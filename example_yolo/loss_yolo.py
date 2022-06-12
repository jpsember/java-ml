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
    if JG.ISSUE42:
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
    #pr("shape of current:",current.shape)

    # Reshape the target to match the current's shape
    target = target.view(current.shape)
    #pr("shape of target:",current.shape)

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

    if include_confidence:
      loss_confidence = self.construct_confidence_loss(true_confidence, pred_objectness)
      #self.log_tensor("loss_confidence")
      loss = loss + loss_confidence

    if include_objectness:
      iou_scores = self.calculate_iou(ground_cxcy, ground_wh, pred_cxcy, pred_wh)
      self.log_tensor(".iou_scores")
      loss_objectness = self.construct_objectness_loss(true_confidence, pred_objectness, iou_scores)
      self.log_tensor(".loss_objectness")
      loss = loss + loss_objectness

    if not warning("disabled class_loss"):
      true_class_probabilities = target[:, :, :, F_CLASS_PROBABILITIES:class_prob_end]
      # Determine each predicted box's set of conditional class probabilities.
      #
      predicted_box_class_logits = current[:, :, :, F_CLASS_PROBABILITIES:class_prob_end]
      loss_class = self.construct_class_loss(true_confidence, true_class_probabilities, predicted_box_class_logits)
      loss = loss + loss_class

    if False and warning("just using iou_scores as loss"):
      t_iou = iou_scores.view(coord_mask.shape)
      loss = (1.0 - t_iou) * coord_mask
      loss = loss.sum()
      self.log_tensor(".loss:1")

    #pr("shape of loss:", loss.shape)
    
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



  def construct_objectness_loss(self, true_confidence, pred_objectness, iou_scores):
    #
    # true_confidence is ZERO where there are no true boxes.
    #
    self.log_tensor(".true_confidence")
    self.log_tensor(".iou_scores")
    self.log_tensor(".pred_objectness")

    true_objectness = true_confidence * iou_scores
    self.log_tensor("true_objectness")

    # The noobj weight should be distributed among all the boxes (anchors * grid cells),
    # otherwise the weight is dependent upon the grid size
    #
    noobj_weight = self.yolo.lambda_noobj
    warning("reducing no obj weight by 10")
    noobj_weight *= 0.1

    if not warning("not scaling objectness loss by box count"):
      boxes_in_grid = self.num_anchors * self.grid_cell_total
      noobj_weight = noobj_weight / boxes_in_grid

    squared_diff = torch.square(true_objectness - pred_objectness)
    self.log_tensor(".squared_diff")

    # conf_mask is the sum of two terms, at least one of which must be zero since
    # one is multiplied by (1 - true_box_exists), and the other by (true_box_exists),
    # where true_box_exists is an indicator variable (either 0 or 1).
    #
    # Each conf_mask element, if nonzero, is a multiplier to apply to the difference in
    # true and predicted confidences for a box.
    #
    # If a conf_mask element is nonzero, that means either:
    #
    #    i) there *is* an object there, and we expect a high predicted confidence;
    # or,
    #   ii) there *isn't* an object there, and we expect a low predicted confidence.
    #

    # It is clear that each element of conf_mask is either obj_scale or no_obj_scale,
    # so the number of nonzero entries below is just the number of anchor boxes in the entire image...

    no_obj_mask = 1.0 - true_confidence
    conf_mask = no_obj_mask * noobj_weight + true_confidence
    self.log_tensor(".conf_mask")

    masked_diff = squared_diff * conf_mask
    self.log_tensor(".masked_diff")
    return torch.sum(masked_diff)



  def construct_class_loss(self, true_confidence, true_class_probabilities, predicted_logits):
    #
    # arg_max:  Returns the index with the largest value across dimensions of a tensor
    #
    # Gets index of highest true conditional class probability; i.e. converts from one-hot to index.
    #
    true_box_class = torch.argmax(true_class_probabilities, dim=-1)   # -1 means the last dimension
    show("true_class_prob",true_class_probabilities)
    show("with argmax",true_box_class)


    # I suspect this is to apply a nonuniform weighting to individual classes.
    # At present, all classes (categories) have unit weight, so it has no effect...?
    #
    # Or: it takes a tensor representing a class index (true_box_class) and constructs a 'one-hot' slice from it...?
    #
    class_wt = torch.ones(self.yolo.category_count, dtype=torch.float32)
    show("class_wt",class_wt)
    show("true_box_class",true_box_class)

    halt("need to figure out 'gather'")

    _tmp = torch.gather(class_wt, true_box_class)
    show("gather",_tmp)
    halt()

    class_mask = true_confidence * _tmp
    nb_class_box = tf.maximum(1., tf.reduce_sum(input_tensor=tf.cast(class_mask > 0.0, dtype=tf.float32)))

    _tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=predicted_logits)
    _tmp = tf.reduce_sum(input_tensor=_tmp * class_mask) / nb_class_box

    return _tmp


