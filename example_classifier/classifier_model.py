from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork

# The model for the classifier_train.py program
#
class ClassifierModel(JsModel):


  def __init__(self, network: NeuralNetwork):
    super(ClassifierModel, self).__init__(network)


