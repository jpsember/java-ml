from pycore.pytorch_util import *
from gen.yolo import Yolo
from .yolo_util import *

class YoloLayer(nn.Module):

  def __init__(self, yolo:Yolo):
    super(YoloLayer, self).__init__()
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)
    self.grid_size = grid_size(yolo)
    self.grid_cell_total = self.grid_size.product()


  def forward(self, current):

    # Apply narrowing functions to appropriate fields now, so we don't need to do them in the loss function or the
    # Java code
    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    y = self.yolo

    batch_size = current.data.size(0)
    current = current.view(batch_size, self.grid_cell_total, self.num_anchors, -1)  # -1 : infer remaining

    class_prob_end = F_CLASS_PROBABILITIES + y.category_count

    # Determine predicted box's x,y
    #
    # Apply narrowing transformation to map (-inf...+inf) to (0...1): sigmoid function
    #
    pred_cxcy = torch.sigmoid(current[:, :, :, F_BOX_CX:F_BOX_CY + 1])

    # Determine each predicted box's w,h
    #
    # Apply narrowing transformation to map (-inf...+inf) to (0..+inf): exp function
    #
    pred_wh = my_exp(current[:, :, :, F_BOX_W:F_BOX_H+1])

    # Determine each predicted box's confidence score.
    # Apply narrowing transformation to map (-inf...+inf) to (0..1): sigmoid function
    #
    pred_objectness = torch.sigmoid(current[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1])

    # Determine each predicted box's class probabilities.
    # Apply narrowing transformation to map (-inf...+inf) to (0..1): sigmoid function
    #
    # ....actually, see issue #59; does pytorch expect 'raw' logits?
    #pred_categories = torch.sigmoid(current[:, :, :, F_CLASS_PROBABILITIES:class_prob_end])
    pred_categories = current[:, :, :, F_CLASS_PROBABILITIES:class_prob_end]

    # Concatenate the modified bits together into another tensor
    # TODO: can we apply the above mappings 'in-place' to avoid this step?
    x = torch.cat((pred_cxcy, pred_wh, pred_objectness, pred_categories), D_BOXINFO)

    return x
