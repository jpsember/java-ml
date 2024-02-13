import torch.nn.functional

from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.tensor_logger import TensorLogger
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

    pr("batch size:",self.batch_size)

    pr("current:",current)
    print(current)
    # This has structure
    #
    # [image, category prob]
    #

    # The tensor we want to feed into the crossentropyloss is called 'outputs' in the other tutorial,
    # and has this structure:
    #
    #  [image, class log prob],
    #  e.g.  [[ 5.8728, -4.8610,  3.1049, -0.7258,  3.2347,  0.1869, -1.9577, -0.8296, 1.4538, -3.0592],
    #         [ 1.2802,  1.8693, -1.5234, -1.1094, -1.6835, -1.6081, -2.5142, -0.6030, 1.7281,  3.8946],
    #         [-0.1654, -2.9004, -1.0536,  1.4381,  2.7306,  2.9377, -3.5011,  6.4039, -4.6020, -0.5139],
    #         [-0.6465, -4.9333,  1.3352, -0.0241,  5.4004,  1.9908, -2.1186, 10.1500, -6.3850, -5.2495]]

    # Reshape the target to match the current's shape
    show("target before reshape:",target)
    # target = target.view(current.shape)
    # show("target after reshape:",target)

    if self.cross_entropy_loss is None:
      # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")



    # The CrossEntropyLoss expects a target of type 'Long', not float
    classification_loss = self.cross_entropy_loss(current, target.type(torch.LongTensor) )
    show("classification_loss", classification_loss)

    self.log_tensor("classification_loss")
    #See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    # We sum a loss tensor's components to a single (scalar) value.
    # Give the resulting tensors the prefix 'scalar_' to denote this.

    scalar_classification = classification_loss.sum()

    if include_aux_stats:
      self.add_aux_stat("loss_class", scalar_classification)

    scalar_loss = scalar_classification
    return scalar_loss / self.batch_size


  # Calculate loss component from a tensor and store in the aux_stats dict.
  # If tensor is None, does nothing
  #
  def add_aux_stat(self, key, tensor):
    if tensor is not None:
      self.aux_stats[key] = tensor.item() / self.batch_size


  # Send a tensor for logging
  #
  def log_tensor(self, name, t=None):
    warning("doing nothing yet; this was yolo logging")
    #TensorLogger.default_instance.report_grid(t, name, size=self.grid_size)
