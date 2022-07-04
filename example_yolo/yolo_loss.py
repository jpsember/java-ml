from gen.yolo import Yolo
from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.tensor_logger import TensorLogger
from .yolo_util import *


def squared_difference(a, b):
  return (a - b) ** 2

class YoloLoss(nn.Module):

  def __init__(self, network: NeuralNetwork, yolo:Yolo):
    super(YoloLoss, self).__init__()
    self.network = network
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)
    self.grid_size = grid_size(yolo)
    self.grid_cell_total = self.grid_size.product()
    self.log_counter = 0
    self.logged_tensor_count = 0

  def forward(self, current, target):

    EPSILON = 1e-8

    self.log_counter += 1
    yolo = self.yolo
    batch_size = current.data.size(0)

    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    current = current.view(batch_size, self.grid_cell_total, self.num_anchors, -1)  # -1 : infer remaining

    # Reshape the target to match the current's shape
    target = target.view(current.shape)

    ground_confidence = target[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1]

    ground_cxcy = target[:, :, :, F_BOX_CX:F_BOX_CY + 1]
    ground_wh   = target[:, :, :, F_BOX_W:F_BOX_H + 1]
    self.log_tensor(".ground_cxcy")
    self.log_tensor(".ground_wh")

    # TODO: classification loss
    #class_prob_end = F_CLASS_PROBABILITIES + y.category_count

    pred_cxcy = current[:, :, :, F_BOX_CX:F_BOX_CY+1]
    self.log_tensor(".pred_cxcy")

    pred_wh = current[:, :, :, F_BOX_W:F_BOX_H+1]
    self.log_tensor(".pred_wh")



    # Calculate generalized IoU
    #


    # Predicted x, y, width, height
    #
    pred_width = pred_wh[:,:,:,0:1]
    pred_height = pred_wh[:,:,:,1:2]

    pred_width_half = pred_width * 0.5
    pred_height_half = pred_height * 0.5

    pred_cx = pred_cxcy[:,:,:, 0:1]
    pred_x1 = pred_cx - pred_width_half
    pred_x2 = pred_cx + pred_width_half

    pred_cy = pred_cxcy[:,:,:, 1:2]
    pred_y1 = pred_cy - pred_height_half
    pred_y2 = pred_cy + pred_height_half


    # Ground x, y, width, height
    #
    ground_width = ground_wh[:, :, :, 0:1]
    ground_height = ground_wh[:, :, :, 1:2]

    ground_width_half = ground_width * 0.5
    ground_height_half = ground_height * 0.5

    ground_cx = ground_cxcy[:, :, :, 0:1]
    ground_x1 = ground_cx - ground_width_half
    ground_x2 = ground_cx + ground_width_half

    ground_cy = ground_cxcy[:, :, :, 1:2]
    ground_y1 = ground_cy - ground_height_half
    ground_y2 = ground_cy + ground_height_half


    pred_area = pred_width * pred_height
    ground_area = ground_width * ground_height

    # Intersection between predicted and ground boxes
    #
    x1_inter = torch.maximum(pred_x1, ground_x1)
    x2_inter = torch.minimum(pred_x2, ground_x2)
    y1_inter = torch.maximum(pred_y1, ground_y1)
    y2_inter = torch.minimum(pred_y2, ground_y2)

    intersection_area = torch.clamp((x2_inter - x1_inter), min=0.0) \
            * torch.clamp((y2_inter - y1_inter), min=0.0)

    # Find coordinates of smallest enclosing box Bc:
    #
    x1c = torch.minimum(pred_x1, ground_x1)
    x2c = torch.maximum(pred_x2, ground_x2)
    y1c = torch.minimum(pred_y1, ground_y1)
    y2c = torch.maximum(pred_y2, ground_y2)

    self.log_tensor(".pred_x1")
    self.log_tensor(".pred_x2")
    self.log_tensor(".pred_y1")
    self.log_tensor(".pred_y2")

    # Calculate area of Bc:
    #
    container_area = (x2c - x1c) * (y2c - y1c)


    # Calculate IoU
    #
    union_area = pred_area + ground_area - intersection_area

    iou = intersection_area / (union_area + EPSILON)
    self.log_tensor(".iou")

    # Let's normalize the giou so that it is between 0 and 1
    #
    self.log_tensor(".union_area")
    self.log_tensor(".container_area")

    giou = iou - ((container_area - union_area) / (container_area + EPSILON))
    self.log_tensor(".giou")
    self.log_tensor(".iou")

    norm_giou = (1.0 + giou) * 0.5
    self.log_tensor(".norm_giou")

    self.log_tensor("iou", iou * ground_confidence)
    self.log_tensor("norm_giou", norm_giou * ground_confidence)

    pred_objectness = current[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1]
    self.log_tensor("pred_objectness")

    # loss_box_pos is loss for inaccurately predicted ground object box positions
    #
    loss_box_pos = (ground_confidence * (1.0 - norm_giou)) * yolo.lambda_coord
    self.log_tensor("loss_box_pos")

    # loss_objectness_box :  loss for inaccurately predicting objectness when a ground box *exists*
    # loss_objectness_nobox :  loss for inaccurately predicting objectness when a ground box *doesn't exist*
    #
    loss_objectness_box = ground_confidence * squared_difference(norm_giou, pred_objectness)
    loss_objectness_nobox = (1 - ground_confidence) * pred_objectness * yolo.lambda_noobj

    self.log_tensor("loss_objectness_box")
    self.log_tensor("loss_objectness_nobox")

    loss = (loss_box_pos + loss_objectness_box + loss_objectness_nobox).sum() / batch_size
    return loss


  def log_active(self) -> bool:
    return self.log_counter % 5 == 0


  # Send a tensor for logging.  Assumes it has the dimension D_IMAGE, D_GRIDSIZE, etc
  #
  def log_tensor(self, name, t=None):
    if not self.log_active():
      return
    if name.startswith("."):
      return
    if self.logged_tensor_count >= 30:
      return
    self.logged_tensor_count += 1

    # If tensor not provided, assume name refers to a local variable in the caller's scope
    #
    t = get_var(t, name)

    # Construct a slice of the tensor for inspection
    z = t.detach()
    if len(z.size()) == 0:
      TensorLogger.default_instance.add_msg(f"{name}: {z.data:5.3}")
      return


    first_page_only = True
    if False and name == "masked_diff":
      first_page_only = False

    height = self.grid_size.y
    width = self.grid_size.x

    if first_page_only:
      z = z[0,:]
      z = z.view(height,width,-1)
    else:
      y = list(z.shape)
      z = z.view(y[0],height,width,-1)

    if first_page_only and width > 8:
      # Zoom in on the center grid cells
      #     ROWS COLS
      z = z[4:7, 5:8,:]
    TensorLogger.default_instance.add(z, name)
