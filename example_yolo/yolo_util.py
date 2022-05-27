from gen.yolo import Yolo
from pycore.ipoint import IPoint
from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork

# Define some constants
#

F_BOX_XYWH = 0
F_CONFIDENCE = F_BOX_XYWH + 4
F_CLASS_PROBABILITIES = F_CONFIDENCE + 1

def anchor_box_count(yolo : Yolo) -> int:
  todo("The generated python code for 'list of IPoints' is doing type inference as the scalar, not the list")
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
