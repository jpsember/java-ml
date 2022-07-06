from gen.yolo import Yolo
from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.tensor_logger import TensorLogger
from .yolo_util import *
from pycore.jg import JG

class YoloLoss(nn.Module):

  def __init__(self, network: NeuralNetwork, yolo:Yolo):
    super(YoloLoss, self).__init__()
    self.network = network
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)
    self.grid_size = grid_size(yolo)
    self.grid_cell_total = self.grid_size.product()
    self.log_counter = 0
    self.cross_entropy_loss = None


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

    ground_obj_t_mask = target[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1]
    ground_obj_f_mask = (1.0 - ground_obj_t_mask)
    self.log_tensor(".ground_obj_t_mask")
    self.log_tensor(".ground_obj_f_mask")


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
    giou = iou - ((container_area - union_area) / (container_area + EPSILON))

    # Normalize the giou so that it is between 0 and 1
    #
    norm_giou = (1.0 + giou) * 0.5

    # If logging, we should mask out cells where there are no ground truth boxes
    #
    self.log_tensor("iou", iou * ground_obj_t_mask)
    self.log_tensor("norm_giou", norm_giou * ground_obj_t_mask)

    pred_objectness = current[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1]
    self.log_tensor("pred_objectness")


    # Let's add the position and dimensions error back in

    loss_box_center = (squared_difference(pred_cx, ground_cx) + squared_difference(pred_cy, ground_cy)) * ground_obj_t_mask
    loss_box_size   = (squared_difference(pred_width, ground_width) + squared_difference(pred_height, ground_height)) * ground_obj_t_mask
    self.log_tensor("loss_box_center")
    self.log_tensor("loss_box_size")

    if False:
      # loss_box_pos is loss for inaccurately predicted ground object box positions
      #
      loss_box_pos = (ground_obj_t_mask * (1.0 - norm_giou)) * yolo.lambda_coord
      self.log_tensor("loss_box_pos")

    # loss_objectness_box :  loss for inaccurately predicting objectness when a ground box *exists*
    # loss_objectness_nobox :  loss for inaccurately predicting objectness when a ground box *doesn't exist*
    #
    loss_objectness_box   = ground_obj_t_mask * squared_difference(norm_giou, pred_objectness)
    loss_objectness_nobox = ground_obj_f_mask * pred_objectness
    self.log_tensor("loss_objectness_box")
    self.log_tensor("loss_objectness_nobox")

    loss = (  loss_box_center * yolo.lambda_coord \
            + loss_box_size * yolo.lambda_coord \
            + loss_objectness_box \
            + loss_objectness_nobox * yolo.lambda_noobj )

    # Include a classification loss if there is more than a single category
    #
    if yolo.category_count > 1:
      ground_category_onehot = target[:, :, :, F_CLASS_PROBABILITIES:F_CLASS_PROBABILITIES + yolo.category_count]
      #self.log_tensor("ground_category_onehot")

      # We need to cast to a float if we want the logger to handle it
      #self.log_tensor("ground_box_class", ground_box_class.type(torch.FloatTensor))

      pred_class = current[:, :, :, F_CLASS_PROBABILITIES:F_CLASS_PROBABILITIES + yolo.category_count]
      # Note: our logging collapses some of the dimensions, so the n probabilities all appear to 'stretch out' the displayed width
      #self.log_tensor("pred_class")

      if self.cross_entropy_loss is None:
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

      # I think we need to reshape the input and target using views
      # so the 'minibatch' includes all the probability records, e.g.
      # images * cells * anchors...

      input_view = pred_class.view(-1, yolo.category_count)
      target_view = ground_category_onehot.view(-1, yolo.category_count)

      ce_loss_view = self.cross_entropy_loss(input_view, target_view)

      # Reshape the loss so we again have results for each image, cell, anchor...
      #
      img_count, cell_count, anchor_count, _ = pred_class.shape
      classificiation_loss = ce_loss_view.view(img_count,cell_count,anchor_count,-1)

      self.log_tensor("classificiation_loss")
      #See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

      loss = loss + classificiation_loss


    loss = loss.sum() / batch_size

    if loss.data > 2000:
      die("Loss has ballooned to:", loss.data)
    return loss


  # Send a tensor for logging
  #
  def log_tensor(self, name, t=None):
    if False and warning("omitting logging tensor"):
      return
    TensorLogger.default_instance.report_grid(t, name, size=self.grid_size)
