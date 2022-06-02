import torch

from gen.yolo import Yolo
from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork
from pycore.tensor_logger import TensorLogger
from .yolo_util import *


class YoloLoss(nn.Module):

  def __init__(self, network: NeuralNetwork, yolo:Yolo):
    super(YoloLoss, self).__init__()
    self.logger = TensorLogger()
    self.network = network
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)
    self.grid_size = grid_size(yolo)
    self.grid_cell_total = self.grid_size.product()
    self.anchors = self.construct_anchors_tensor()


  # Construct a tensor containing the anchor boxes, normalized to grid cell space
  #
  def construct_anchors_tensor(self):
    yolo = self.yolo
    c = []
    b_x, b_y = 1.0 / yolo.block_size.x, 1.0 / yolo.block_size.y
    for box_pixels in yolo.anchor_boxes_pixels:
      c.append((box_pixels.x * b_x, box_pixels.y * b_y))
    anchors = torch.Tensor(c)
    #self.logger.add(anchors, "anchor_boxes (normalized to grid cell)")
    return anchors


  def forward(self, current, target):
    y = self.yolo
    batch_size = current.data.size(0)

    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    current = current.view(batch_size, self.grid_cell_total, self.num_anchors, -1)  # -1 : infer remaining

    # Reshape the target to match the current's shape
    target = target.view(current.shape)

    true_xy = target[:, :, :, F_BOX_X:F_BOX_Y+1]
    self.log_tensor(".true_xy")

    # true_box_wh will be the width and height of the box, relative to the anchor box
    #
    true_wh = target[:, :, :, F_BOX_W:F_BOX_H+1]
    self.log_tensor(".true_wh")

    true_confidence = target[:, :, :, F_CONFIDENCE]
    # Add a dimension to true_confidence so it has equivalent dimensionality as true_xy, true_wh, etc
    # (this doesn't change its volume, only its dimensions) <-- explain?
    #
    # This produces a mask value which we apply to the xy and wh loss.
    coord_mask = true_confidence[..., None]

    class_prob_end = F_CLASS_PROBABILITIES + y.category_count
    true_class_probabilities = target[:, :, :, F_CLASS_PROBABILITIES:class_prob_end]

    # We could have stored the true class number as an index, instead of a one-hot vector;
    # but the symmetry of the structure of the true vs inferred data keeps things simple.

    # Get the number of ground truth boxes in the batch, as a float, and to avoid divide by zero, assume at least one
    #
    num_true_boxes = float(max(1, true_confidence.count_nonzero()))

    just_confidence_logits = current[:, :, :, F_CONFIDENCE]

    # Determine predicted box's x,y
    #
    # We need to map (-inf...+inf) to (0...1); hence apply sigmoid function
    #
    pred_xy = torch.sigmoid(current[:, :, :, F_BOX_X:F_BOX_Y+1]) * coord_mask
    self.log_tensor(".pred_xy")

    # Determine each predicted box's w,h
    #
    # We need to map (-inf...+inf) to (0..+inf); hence apply the exp function
    #
    pred_wh = torch.exp(current[:, :, :, F_BOX_W:F_BOX_H+1]) * coord_mask
    self.log_tensor(".pred_wh")

    # Determine each predicted box's confidence score.
    # We need to map (-inf...+inf) to (0..1); hence apply sigmoid function
    #
    predicted_confidence = torch.sigmoid(just_confidence_logits)

    # Determine each predicted box's set of conditional class probabilities.
    #
    predicted_box_class_logits = current[:,:,:,F_CLASS_PROBABILITIES:class_prob_end]

    if False:
      show_shape("coord_mask")
      show_shape("true_confidence")
      show_shape("true_xy")
      show_shape("pred_xy")

    x = (true_xy - pred_xy).square()
    #show_shape("true-pred squared",x)

    self.log_tensor(".xy true-pred", x)

    # TODO: why can't we just set the 'box' loss based on the IOU inaccuracy?  Then
    # presumably the x,y,w,h will naturally move to the target?
    loss_xy = x.sum().item() / num_true_boxes

    _tmp = ((true_wh - pred_wh).square())
    show(".wh error", _tmp)
    loss_wh = _tmp.sum().item() / num_true_boxes

    iou_scores = self.calculate_iou(true_xy, true_wh, pred_xy, pred_wh)
    self.log_tensor("iou_scores")

    loss_confidence = self.construct_confidence_loss(true_confidence, iou_scores, predicted_confidence)
    show(".loss_confidence:",loss_confidence)

    _coord_scaled = self.yolo.lambda_coord * (loss_xy + loss_wh)
    show("._coord_scaled", _coord_scaled)

    _tmp = _coord_scaled + loss_confidence

    if not warning("disabled class_loss"):
      loss_class = self.construct_class_loss(true_confidence, true_class_probabilities, predicted_box_class_logits)
      _tmp = _tmp + loss_class

    _tmp = _tmp.mean()
    return _tmp



  # Send a tensor for logging.  Assumes it has the dimension D_IMAGE, D_GRIDSIZE, etc
  #
  def log_tensor(self, name, t=None):
    if name.startswith("."):
      return

    # If tensor not provided, assume name refers to a local variable in the caller's scope
    #
    t = get_var(t, name)

    # Construct a slice of the tensor for inspection
    z = t.detach()
    z = z[0,:]

    height = self.grid_size.y
    width = self.grid_size.x

    z = z.view(height,width,-1)
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
    self.log_tensor("true_offset")
    self.log_tensor("true_box_min")
    self.log_tensor("true_box_max")
    halt()

    # Calculate the min/max extents of the predicted boxes
    #
    pred_offset = pred_wh / 2.
    self.log_tensor("pred_wh")
    pred_box_min = pred_xy - pred_offset
    pred_box_max = pred_xy + pred_offset
    show(".pred_box_min", pred_box_min)
    show(".pred_box_max", pred_box_max)

    self.log_tensor("pred_box_min")
    self.log_tensor("pred_box_max")




    # Determine the area of their intersection (which may be zero)
    #
    intersect_box_min = torch.maximum(true_box_min, pred_box_min)
    intersect_box_max = torch.minimum(true_box_max, pred_box_max)

    intersect_box_size = torch.clamp(intersect_box_max - intersect_box_min, min=0.0)
    intersect_box_area = intersect_box_size[..., 0] * intersect_box_size[..., 1]

    true_box_area = true_wh[..., 0] * true_wh[..., 1]
    predicted_box_area = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = predicted_box_area + true_box_area - intersect_box_area

    # For cells that have no ground truth box, the area will be zero; so to avoid a possible divide by zero
    # (which may be harmless, but will be confusing), add an epsilon to the denominator
    #
    return torch.div(intersect_box_area, torch.clamp(union_areas, min=1e-8))



  def construct_confidence_loss(self, true_confidence, iou_scores, predicted_confidence):
    true_box_confidence = iou_scores * true_confidence

    _zeros = torch.zeros_like(true_confidence)
    _ones = torch.ones_like(_zeros)
    _no_objects_expected = torch.where(torch.greater(true_confidence, 0.0), _zeros, _ones)

    conf_mask = _no_objects_expected * self.yolo.lambda_noobj + true_confidence

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


    anchor_boxes_per_image = self.num_anchors * self.grid_cell_total
    loss_conf = torch.sum(torch.square(true_box_confidence - predicted_confidence) * conf_mask) / anchor_boxes_per_image
    return loss_conf



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


