from gen.yolo import Yolo
from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork


class YoloModel(JsModel):


  def __init__(self, network: NeuralNetwork):
    super(YoloModel, self).__init__(network)
    warning("the network arg doesn't seem to be required")


  def process_custom_layer(self, lyr):
    if lyr.type != "yolo":
      die("unsupported layer type:", lyr.type)
    if lyr.input_volume != lyr.output_volume:
      self.add_layer(nn.Flatten(), "fc.Flatten")
      self.add_layer(nn.Linear(vol_volume(lyr.input_volume), vol_volume(lyr.output_volume)), "fc.Linear")
    else:
      die("untested if input volume = output volume")
