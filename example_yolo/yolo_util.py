from gen.yolo import Yolo
from pycore.ipoint import IPoint

# Define some constants
#

# 'Normalized to S' means value has been scaled so that Smin, Smax = 0, 1 resp.
#
F_BOX_XYWH = 0

# center of box, normalized to grid cell bounds
#
F_BOX_CX = F_BOX_XYWH
F_BOX_CY = F_BOX_XYWH + 1

# width and height of box, normalized to anchor box size
#
F_BOX_W = F_BOX_XYWH + 2
F_BOX_H = F_BOX_XYWH + 3

# Input: 0 if no object exists, 1 otherwise
# Output: this represents 'objectness', and ranges 0..1
#
F_CONFIDENCE = F_BOX_XYWH + 4

# Input: one-hot vector; all are 0 except the object's class, which is 1
# Output: largest value represents predicted object's class
#
F_CLASS_PROBABILITIES = F_CONFIDENCE + 1

# Tensor dimensions
#
D_IMAGE = 0
D_GRIDCELL = 1
D_ANCHOR = 2
D_BOXINFO = 3
D_TOTAL = 4


def anchor_box_count(yolo : Yolo) -> int:
  return len(yolo.anchor_boxes_pixels)


def values_per_anchor_box(yolo : Yolo):
  return F_CLASS_PROBABILITIES + yolo.category_count


def values_per_block(yolo : Yolo) -> int:
    return values_per_anchor_box(yolo) * anchor_box_count(yolo)


def grid_size(yolo : Yolo) -> IPoint:
  bs = yolo.block_size
  image_size = yolo.image_size
  grid = IPoint.with_x_y(image_size.x // bs.x, image_size.y // bs.y)
  return grid


def image_label_float_count(yolo : Yolo) -> int:
  return values_per_block(yolo) * grid_size(yolo).product()
