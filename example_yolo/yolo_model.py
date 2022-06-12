from gen.yolo import Yolo
from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork


class YoloModel(JsModel):

  def __init__(self, network: NeuralNetwork):
    super(YoloModel, self).__init__(network)

