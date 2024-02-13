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

    # ...I adapted this code from yolo_loss.py

    # Log some additional stats about the loss values
    #
    include_aux_stats = (JG.batch_number == 0)
    if include_aux_stats:
      self.aux_stats = {}
      JG.aux_stats = self.aux_stats

    self.log_counter += 1
    self.batch_size = current.data.size(0)


    todo("I think I need to convert the category label (scalar) to a one-hot vector")

    current = current.view(self.batch_size)  # -1 : infer remaining
    show("current:",current)
    current = torch.nn.functional.one_hot(current.long(), self.categories)
    show("current after one hot:", current)
    pr(str(current))

    # Reshape the target to match the current's shape
    show("target before reshape:",target)
    target = target.view(current.shape)
    show("target after reshape:",target)

    pr("self.categories:",self.categories)
    ground_category_onehot = target[:, 0:self.categories]
    show("ground_category_onehot",ground_category_onehot)

    pred_class = current[:,0:self.categories]
    show("pred_class", pred_class)

    pr("pred_class:",pred_class)

    if self.cross_entropy_loss is None:
      # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    # We need to reshape the input and target using views     (OLD YOLO COMMENT)
    # so the 'minibatch' includes all the probability records, e.g.
    # images * cells * anchors...
    input_view = pred_class.view(-1, self.categories)
    show("input_view", input_view)
    target_view = ground_category_onehot.view(-1, self.categories)
    show("target_view", target_view)
    ce_loss_view = self.cross_entropy_loss(input_view, target_view)
    show("ce_loss_view", ce_loss_view)

    # Reshape the loss so we again have results for each image, cell, anchor...  (OLD YOLO COMMENT)
    #
    img_count,  _ = pred_class.shape
    classification_loss = ce_loss_view.view(img_count,-1)

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
