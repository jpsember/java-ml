import torch.nn.functional

from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.jg import JG
from gen.classifier import Classifier

class ClassifierLoss(nn.Module):

  def __init__(self, network: NeuralNetwork, classifier:Classifier):
    super(ClassifierLoss, self).__init__()
    self.classifier = classifier
    self.network = network
    self.log_counter = 0
    self.cross_entropy_loss = None
    self.batch_size = None
    self.aux_stats = None
    self.categories = classifier.category_count
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    self.cross_entropy_loss = nn.CrossEntropyLoss()   # (reduction="none")

  def forward(self, current, target):

    # Log some additional stats about the loss values
    #
    include_aux_stats = (JG.batch_number == 0)
    if include_aux_stats:
      self.aux_stats = {}
      JG.aux_stats = self.aux_stats

    self.log_counter += 1
    self.batch_size = current.data.size(0)

    check_state(target.dtype == torch.long,"target dtype is not long")
    classification_loss = self.cross_entropy_loss(current, target)

    # show("classification_loss", classification_loss)

    #See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    return classification_loss


  # Calculate loss component from a tensor and store in the aux_stats dict.
  # If tensor is None, does nothing
  #
  # (Not used at present!)
  #
  def add_aux_stat(self, key, tensor):
    if tensor is not None:
      self.aux_stats[key] = tensor.item() / self.batch_size

