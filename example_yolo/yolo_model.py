from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork


class YoloModel(JsModel):


  def __init__(self, network: NeuralNetwork):
    super(YoloModel, self).__init__(network)


  def process_custom_layer(self, lyr):
    if lyr.type != "yolo":
      die("unsupported layer type:", lyr.type)
    pr("layer:")
    pr(lyr)
    # If the input depth differs from the output depth, 
    die("not finished")


