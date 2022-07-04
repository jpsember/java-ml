from gen.train_param import TrainParam
from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork
from .yolo_layer import YoloLayer
from .yolo_util import *

class YoloModel(JsModel):

  def __init__(self, network: NeuralNetwork, yolo:Yolo):
    super(YoloModel, self).__init__(network)
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)
    self.grid_size = grid_size(yolo)
    self.grid_cell_total = self.grid_size.product()


  def process_custom_layer(self, lyr):
    if lyr.type != "yolo":
      die("unsupported layer type:", lyr.type)

    if lyr.input_volume != lyr.output_volume:
      self.add_layer(nn.Flatten(), "fc.Flatten")
      self.add_layer(nn.Linear(vol_volume(lyr.input_volume), vol_volume(lyr.output_volume)), "fc.Linear")
    else:
      die("untested if input volume = output volume")

    self.add_layer(YoloLayer(self.yolo))
