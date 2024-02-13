from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork

# The model for the classifier_train.py program
#
class ClassifierModel(JsModel):


  def __init__(self, network: NeuralNetwork):
    super(ClassifierModel, self).__init__(network)
    self.categories = 2
    todo("categories is a constant, 2, until I refactor method of passing classifier information around, e.g. categories")


  def process_custom_layer(self, lyr):
    if lyr.type != "classifier":
      die("unsupported layer type:", lyr.type)

    ncat = self.categories

    # Flatten the input volume into a fibre, then apply a linear layer
    #
    # See: https://stackoverflow.com/a/60372416
    #
    # Flatten:  https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    #
    ly = self.layer
    self.add_layer(nn.Flatten(), "fc.Flatten")
    check_state(ncat > 1, "attempt to construct fc layer with ncat=", ncat)
    self.add_layer(nn.Linear(vol_volume(ly.input_volume), ncat), "fc.Linear")
