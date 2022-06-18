from gen.yolo import Yolo
from gen.log_item import *
from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.tensor_logger import TensorLogger
from .yolo_util import *


class YoloLoss(nn.Module):

  def __init__(self, network: NeuralNetwork, logger:TensorLogger, yolo:Yolo):
    super(YoloLoss, self).__init__()
    self.logger = logger
    self.network = network
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)
    self.grid_size = grid_size(yolo)
    self.grid_cell_total = self.grid_size.product()
    self.log_counter = 0


  def forward(self, current, target):
    verify_not_nan("YoloLoss.forward", "current")

    include_objectness = False
    include_confidence = True

    self.log_counter += 1
    #if include_objectness and self.log_active():
    #  self.logger.add_msg("^v;loss_yolo.py\n^d;")
    y = self.yolo
    batch_size = current.data.size(0)

    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    current = current.view(batch_size, self.grid_cell_total, self.num_anchors, -1)  # -1 : infer remaining

    # Reshape the target to match the current's shape
    target = target.view(current.shape)

    ground_confidence = target[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1]

    ground_cxcy = target[:, :, :, F_BOX_CX:F_BOX_CY + 1] * ground_confidence
    self.log_tensor(".ground_cxcy")

    # true_box_wh will be the width and height of the box, relative to the anchor box
    #
    ground_wh = target[:, :, :, F_BOX_W:F_BOX_H+1] * ground_confidence

    # Determine number of ground truth boxes.  Clamp to minimum of 1 to avoid divide by zero.
    #
    true_box_count = torch.clamp(torch.count_nonzero(ground_confidence), min=1).to(torch.float)


    # We could have stored the true class number as an index, instead of a one-hot vector;
    # but the symmetry of the structure of the true vs inferred data keeps things simple.
    #
    class_prob_end = F_CLASS_PROBABILITIES + y.category_count

    pred_cxcy = current[:, :, :, F_BOX_CX:F_BOX_CY + 1] * ground_confidence
    self.log_tensor(".pred_cxcy")

    pred_wh = current[:, :, :, F_BOX_W:F_BOX_H+1] * ground_confidence
    verify_not_nan("loss_yolo_fwd", "pred_wh")
    self.log_tensor(".pred_wh")

    pred_objectness = current[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1]
    self.log_tensor(".pred_objectness")

    lambda_coord = 5.0
    loss_xy = (ground_cxcy - pred_cxcy).square() * lambda_coord

    self.log_tensor(".ground_wh")
    self.log_tensor(".pred_wh")
    verify_non_negative("ground_wh")
    verify_non_negative("pred_wh")

    # FFS, taking sqrt of zero can cause gradient to be NaN;
    #  https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702
    #  
    loss_wh = (torch.sqrt(ground_wh + 1e-8) - torch.sqrt(pred_wh + 1e-8)).square() * lambda_coord
    self.log_tensor(".loss_wh")

    todo("Should we scale the loss function (box etc) by number of anchor boxes & cells?")

    loss_confidence = self.construct_confidence_loss(ground_confidence, pred_objectness)
    self.log_tensor(".ground_confidence")
    self.log_tensor(".pred_objectness")
    self.log_tensor(".loss_confidence")
    #self.log_tensor("loss_confidence")

    boxes_total = self.grid_cell_total + self.num_anchors
    #pr("loss_xy")
    #pr(loss_xy)
    self.log_tensor(".loss_xy")
    loss_xy_sum = loss_xy.sum() / boxes_total
    loss_wh_sum = loss_wh.sum() / boxes_total
    loss_confidence_sum = loss_confidence.sum() / boxes_total


    self.log_tensor(".loss_xy_sum")
    self.log_tensor(".loss_wh_sum")
    self.log_tensor(".loss_confidence_sum")

    loss = loss_xy_sum + loss_wh_sum + loss_confidence_sum
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

    # If tensor not provided, assume name refers to a local variable in the caller's scope
    #
    t = get_var(t, name)

    # Construct a slice of the tensor for inspection
    z = t.detach()
    if len(z.size()) == 0:
      self.logger.add_msg(f"{name}: {z.data:5.3}")
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
    self.logger.add(z, name)


  def calculate_iou(self, true_xy, true_wh, pred_xy, pred_wh):

    # The _xy fields are the box midpoints, and we need to know the edge coordinates

    # Calculate the min/max extents of the true boxes
    #
    true_offset = true_wh / 2.
    true_box_min = true_xy - true_offset
    true_box_max = true_xy + true_offset
    self.log_tensor(".true_box_min:1")
    self.log_tensor(".true_box_max:1")
    true_area = true_wh[..., 0] * true_wh[..., 1]
    self.log_tensor(".true_area:1")

    # Calculate the min/max extents of the predicted boxes
    #
    pred_offset = pred_wh / 2.
    self.log_tensor(".pred_wh:1")
    pred_box_min = pred_xy - pred_offset
    pred_box_max = pred_xy + pred_offset
    self.log_tensor(".pred_box_min:1")
    self.log_tensor(".pred_box_max:1")
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    self.log_tensor(".pred_area:1")

    # Determine the area of their intersection (which may be zero)
    #
    isect_min = torch.maximum(true_box_min, pred_box_min)
    isect_max = torch.minimum(true_box_max, pred_box_max)
    self.log_tensor(".isect_min:1")
    self.log_tensor(".isect_max:1")

    isect_size = torch.clamp(isect_max - isect_min, min=0.0)
    self.log_tensor(".isect_size:1")
    isect_area = isect_size[..., 0] * isect_size[..., 1]
    self.log_tensor(".isect_area:1")

    union_area = pred_area + true_area - isect_area
    self.log_tensor(".union_area:1")

    # For cells that have no ground truth box, the area will be zero; so to avoid a possible divide by zero
    # (which may be harmless, but will be confusing), add an epsilon to the denominator
    #
    iou = torch.div(isect_area, torch.clamp(union_area, min=1e-8))
    self.log_tensor(".iou:1")
    return iou



  def construct_confidence_loss(self, true_confidence, pred_confidence):
    self.log_tensor(".true_confidence")
    self.log_tensor(".pred_confidence")

    squared_diff = torch.square(true_confidence - pred_confidence)
    self.log_tensor(".squared_diff")
    no_obj_mask = 1.0 - true_confidence
    self.log_tensor(".no_obj_mask")
    lambda_noobj = 0.5
    conf_mask = no_obj_mask * lambda_noobj + true_confidence
    self.log_tensor(".conf_mask")
    return squared_diff * conf_mask




