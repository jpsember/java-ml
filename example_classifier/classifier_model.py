from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork

# The model for the classifier_train.py program
#
class ClassifierModel(JsModel):


  def __init__(self, network: NeuralNetwork):
    super(ClassifierModel, self).__init__(network)


  def process_custom_layer(self, lyr):
    if lyr.type != "classifier":
      die("unsupported layer type:", lyr.type)

    if lyr.input_volume != lyr.output_volume:
      check_state(self.layer.filters == 1, "unexpected layer.filters:",self.layer.filters)
      self.construct_fc()
    else:
      die("untested if input volume = output volume")

