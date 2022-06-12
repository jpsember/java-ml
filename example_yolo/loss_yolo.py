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
    JG.issue_42_counter += 1
    pr("issue 42 counter:", JG.issue_42_counter)
    if JG.issue_42_counter == 8:
      li = self.logger.new_log_item()
      li.family_size = 4
      li.family_id = 8000 + JG.issue_42_counter
      li.special_handling = 2
      li.family_slot = 0
      li.message = "images_input"
      self.logger.add(JG.recent_images_input,li)
      li.family_slot = 1
      li.message = "labels_input"
      self.logger.add(JG.recent_labels_input,li)
      li.family_slot = 2
      li.message = "loss_img_input"
      self.logger.add(current,li)
      li.family_slot = 3
      li.message = "loss_lbl_input"
      self.logger.add(target,li)


    verify_not_nan("current")

    self.log_counter += 1
    y = self.yolo
    batch_size = current.data.size(0)

    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    current = current.view(batch_size, self.grid_cell_total, self.num_anchors, -1)  # -1 : infer remaining

    # Reshape the target to match the current's shape
    target = target.view(current.shape)

    ground_cxcy = target[:, :, :, F_BOX_CX:F_BOX_CY + 1]
    self.log_tensor(".ground_cxcy")

    # true_box_wh will be the width and height of the box, relative to the anchor box
    #
    ground_wh = target[:, :, :, F_BOX_W:F_BOX_H+1]
    self.log_tensor(".ground_wh")
    #pr("shape of ground_wh:",ground_wh.shape)

    true_confidence = target[:, :, :, F_CONFIDENCE]

    # Determine number of ground truth boxes.  Clamp to minimum of 1 to avoid divide by zero.
    #
    true_box_count = torch.clamp(torch.count_nonzero(true_confidence), min=1).to(torch.float)

    # Add a dimension to true_confidence so it has equivalent dimensionality as ground_cxcy, ground_wh, etc
    # (this doesn't change its volume, only its dimensions) <-- explain?
    #
    # This produces a mask value which we apply to the xy and wh loss.
    coord_mask = true_confidence[..., None]

    # We could have stored the true class number as an index, instead of a one-hot vector;
    # but the symmetry of the structure of the true vs inferred data keeps things simple.
    #
    class_prob_end = F_CLASS_PROBABILITIES + y.category_count

    # Determine predicted box's x,y
    #
    # We need to map (-inf...+inf) to (0...1); hence apply sigmoid function
    #
    pred_cxcy = torch.sigmoid(current[:, :, :, F_BOX_CX:F_BOX_CY + 1]) * coord_mask
    self.log_tensor(".pred_cxcy")

    # Determine each predicted box's w,h
    #
    # We need to map (-inf...+inf) to (0..+inf); hence apply the exp function
    #
    pred_wh = torch.exp(current[:, :, :, F_BOX_W:F_BOX_H+1]) * coord_mask
    verify_not_nan("pred_wh")
    self.log_tensor(".pred_wh")

    # Determine each predicted box's confidence score.
    # We need to map (-inf...+inf) to (0..1); hence apply sigmoid function
    #
    pred_objectness = torch.sigmoid(current[:, :, :, F_CONFIDENCE])
    self.log_tensor(".pred_objectness")

    x = (ground_cxcy - pred_cxcy).square()
    loss_xy = x.sum() / true_box_count

    self.log_tensor("ground_wh")
    self.log_tensor("pred_wh")
    x = (torch.sqrt(ground_wh) - torch.sqrt(pred_wh)).square()
    self.log_tensor("squareddiff",x)
    loss_wh = x.sum() / true_box_count

    self.log_tensor("loss_wh")
    weighted_box_loss = y.lambda_coord * (loss_xy + loss_wh)
    loss = weighted_box_loss

    todo("Should we scale the loss function (box etc) by number of anchor boxes & cells?")

    loss_confidence = self.construct_confidence_loss(true_confidence, pred_objectness)
    #self.log_tensor("loss_confidence")
    loss = loss + loss_confidence

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



  def construct_confidence_loss(self, true_confidence, pred_confidence):
    self.log_tensor(".true_confidence")
    self.log_tensor(".pred_confidence")
    noobj_weight = self.yolo.lambda_noobj
    squared_diff = torch.square(true_confidence - pred_confidence)
    no_obj_mask = 1.0 - true_confidence
    lambda_obj_weight = 3
    todo("add lambda_obj_weight to yolo params?")
    conf_mask = no_obj_mask * noobj_weight + true_confidence * lambda_obj_weight
    self.log_tensor(".conf_mask")
    masked_diff = squared_diff * conf_mask
    self.log_tensor(".masked_diff")
    return torch.sum(masked_diff)




