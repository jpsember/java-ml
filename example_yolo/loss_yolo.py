import torch

from gen.yolo import Yolo
from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork
from .yolo_util import *

# Derived from https://neptune.ai/blog/pytorch-loss-functions


class YoloLoss(nn.Module):

  def __init__(self, network: NeuralNetwork, yolo:Yolo):
    super(YoloLoss, self).__init__()
    self.network = network
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)
    self.grid_size = grid_size(yolo)
    self.grid_cell_total = self.grid_size.product()


    # Construct a tensor containing the anchor boxes, normalized to the block size
    #
    # Default values, from  https://github.com/uvipen/Yolo-v2-pytorch/blob/master/src/yolo_net.py
    #   anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
    t = []
    bs = yolo.block_size
    b_x, b_y = 1.0 / bs.x, 1.0 / bs.y
    for abp in yolo.anchor_boxes_pixels:
      t.append((abp.x * b_x, abp.y * b_y))
    t = torch.Tensor(t)
    self.anchors = t



  def forward(self, current, target):
    y = self.yolo
    batch_size = current.data.size(0)

    gsize = grid_size(y)
    height = gsize.y
    width = gsize.x
    grid_cell_total = width * height

    # The -1 here makes it inferred from the other dimensions.
    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    current = current.view(batch_size, grid_cell_total, self.num_anchors, -1)

    # Reshape the target to match the current's shape
    target = target.view(current.shape)

    _tmp = pt_to_ftensor(y.image_size)
    _tmp = _tmp.reshape([1, 1, 1, 2])  # Note that the dimension is D_TOTAL
    _image_size = _tmp

    _tmp = pt_to_ftensor(y.block_size)
    _tmp = _tmp.reshape([1, 1, 1, 2])
    _tmp = _tmp / _image_size
    warning("is block_to_image required?")
    _block_to_image = _tmp

    _tmp = self.anchors.reshape([1,1,self.num_anchors,2])
    anchor_wh_img = _tmp
    show(".anchors", anchor_wh_img)
    todo("this is probably NOT the anchor box normalized to image size")

    # Determine ground truth location, size, category

    true_xy_cell = target[:, :, :, F_BOX_X:F_BOX_Y+1]
    show(".true_xy_cell", true_xy_cell)

    # true_box_wh will be the width and height of the box, relative to the anchor box
    #
    true_box_wh = target[:, :, :, F_BOX_W:F_BOX_H+1]
    show(".true_box_wh", true_box_wh)

    true_confidence = target[:, :, :, F_CONFIDENCE]
    show(".true_confidence", true_confidence)

    class_prob_end = F_CLASS_PROBABILITIES + y.category_count

    true_class_probabilities = target[:, :, :, F_CLASS_PROBABILITIES:class_prob_end]  # probably can just do 'x:']
    show(".true_class_probabilities", true_class_probabilities)

    # We could have stored the true class number as an index, instead of a one-hot vector;
    # but the symmetry of the structure of the true vs inferred data keeps things simple.

    # Determine number of ground truth boxes.  Clamp to minimum of 1 to avoid divide by zero.
    # We include any 'neighbor' boxes as well (if there are any).
    #

    # Get the number of ground truth boxes in the batch, as a float, and to avoid divide by zero, assume at least one
    #
    num_true_boxes = float(max(1, true_confidence.count_nonzero()))

    just_confidence_logits = current[:, :, :, F_CONFIDENCE]
    show(".just_confidence_logits", just_confidence_logits)

    # Determine predicted box's x,y
    #
    # We need to map (-inf...+inf) to (0...1); hence apply sigmoid function
    #
    _tmp = current[:, :, :, F_BOX_X:F_BOX_Y+1]
    show(".box x,y",_tmp)
    pred_xy_cell = torch.sigmoid(_tmp)
    show(".pred_xy_cell", pred_xy_cell)

    # Determine each predicted box's w,h
    #
    # We need to map (-inf...+inf) to (0..+inf); hence apply the exp function
    #
    _tmp = current[:, :, :, F_BOX_W:F_BOX_H+1]
    pred_wh_anchor = _tmp
    show(".pred_wh_anchor", pred_wh_anchor)


    # Construct versions of the true and predicted locations and sizes in image units
    #
    true_xy_img = true_xy_cell * _block_to_image
    pred_xy_img = pred_xy_cell * _block_to_image
    warning("do we need the cell coordinates?")
    show(".pred_xy_img", pred_xy_img)


    true_wh_img = true_box_wh * anchor_wh_img
    show(".true_wh_img", true_wh_img)

    pred_wh_img = pred_wh_anchor * anchor_wh_img
    show(".pred_wh_img", pred_wh_img)


    # Determine each predicted box's confidence score.
    # We need to map (-inf...+inf) to (0..1); hence apply sigmoid function
    #
    predicted_confidence = torch.sigmoid(just_confidence_logits)
    show(".predicted_confidence", predicted_confidence)

    # Determine each predicted box's set of conditional class probabilities.
    #
    predicted_box_class_logits = current[:,:,:,F_CLASS_PROBABILITIES:class_prob_end]
    show(".predicted_box_class_logits", predicted_box_class_logits)

    # Add a dimension to true_confidence so it has equivalent dimensionality as true_box_xy, true_box_wh
    # (this doesn't change its volume, only its dimensions)
    #
    # This produces a mask value which we apply to the xy and wh loss.
    # For neighbor box labels, whose confidence < 1, this has the effect of reducing the penalty
    # for those boxes
    #
    _coord_mask = true_confidence[None, :]
    show("._coord_mask", _coord_mask)

    _tmp = (true_xy_cell - pred_xy_cell).square()

    show(".true-pred ^d",_tmp)
    _tmp = _tmp * _coord_mask
    show(".xy true-pred, ^2, * coord_mask", _tmp)
    todo("why does coord_mask have shape [2,2,2,2]?")

    # TODO: why can't we just set the 'box' loss based on the IOU inaccuracy?  Then
    # presumably the x,y,w,h will naturally move to the target?
    loss_xy = _tmp.sum().item() / num_true_boxes

    _tmp = (((true_box_wh - pred_wh_anchor) * _coord_mask).square())
    show(".wh error", _tmp)
    loss_wh = _tmp.sum().item() / num_true_boxes

    iou_scores = self.calculate_iou(true_xy_cell, true_box_wh, pred_xy_cell, pred_wh_anchor)
    show(".iou_scores", iou_scores)

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



  def calculate_iou(self, true_xy, true_wh, pred_xy, pred_wh):

    # The _xy fields are the box midpoints, and we need to know the edge coordinates

    # Calculate the min/max extents of the true boxes
    #
    true_offset = true_wh / 2.
    true_box_min = true_xy - true_offset
    true_box_max = true_xy + true_offset
    show(".true_box_min", true_box_min)
    show(".true_box_max", true_box_max)

    # Calculate the min/max extents of the predicted boxes
    #
    pred_offset = pred_wh / 2.
    pred_box_min = pred_xy - pred_offset
    pred_box_max = pred_xy + pred_offset
    show(".pred_box_min", pred_box_min)
    show(".pred_box_max", pred_box_max)

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



def pt_to_ftensor(pt:IPoint):
  return torch.FloatTensor(pt.tuple())

