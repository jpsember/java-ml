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

    pr("process_custom_layer:",lyr)

    ncat = self.categories

    if lyr.input_volume != lyr.output_volume:
      todo("always assume we are building a layer, regardles of whether the input and output volumes match?")


      #check_state(self.layer.filters == 1, "unexpected layer.filters:",self.layer.filters)
      pr("...constructing fc")

      # We need to 'convert' the input volume into a set of confidences, one for each category.
      #
      # Flatten the input volume into a fibre, then apply a linear layer?
      #
      # See: https://stackoverflow.com/a/60372416
      #
      # Flatten:  https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
      #
      ly = self.layer
      self.add_layer(nn.Flatten(), "fc.Flatten")
      check_state(ncat > 1, "attempt to construct fc layer with ncat=", ncat)
      self.add_layer(nn.Linear(vol_volume(ly.input_volume), ncat), "fc.Linear")
    else:
      die("untested if input volume = output volume")

