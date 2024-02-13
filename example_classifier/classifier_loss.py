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
    check_state(self.categories == 2, "unexpected category count:",classifier)


  def forward(self, current, target):

    # Log some additional stats about the loss values
    #
    include_aux_stats = (JG.batch_number == 0)
    if include_aux_stats:
      self.aux_stats = {}
      JG.aux_stats = self.aux_stats

    self.log_counter += 1
    self.batch_size = current.data.size(0)

    if self.cross_entropy_loss is None:
      # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    check_state(target.dtype == torch.long,"target dtype is not long")
    classification_loss = self.cross_entropy_loss(current, target)
    show(".classification_loss", classification_loss)

    #See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    # We sum a loss tensor's components to a single (scalar) value.
    # Give the resulting tensors the prefix 'scalar_' to denote this.

    scalar_classification = classification_loss.sum()

    if include_aux_stats:
      self.add_aux_stat("loss_classification", scalar_classification)

    scalar_loss = scalar_classification
    return scalar_loss / self.batch_size


  # Calculate loss component from a tensor and store in the aux_stats dict.
  # If tensor is None, does nothing
  #
  def add_aux_stat(self, key, tensor):
    if tensor is not None:
      self.aux_stats[key] = tensor.item() / self.batch_size

