from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.tensor_logger import TensorLogger
from pycore.jg import JG

class ClassifierLoss(nn.Module):

  def __init__(self, network: NeuralNetwork):
    super(ClassifierLoss, self).__init__()
    self.network = network
    self.log_counter = 0
    self.cross_entropy_loss = None
    self.batch_size = None
    self.aux_stats = None
    pr("model config:",network.model_config)
    self.categories = 1
    todo("extract category count from model somehow")


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

    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    # ...for classifier, this simplifies to:
    # Tensor dimensions
    #
    # D_IMAGE = 0
    # D_CLASSPROBS = 1

    current = current.view(self.batch_size, -1)  # -1 : infer remaining

    # Reshape the target to match the current's shape
    target = target.view(current.shape)



    ground_category_onehot = target[:, 0:self.categories]
    pred_class = current[:,0:self.categories]

    if self.cross_entropy_loss is None:
      self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    # We need to reshape the input and target using views     (OLD YOLO COMMENT)
    # so the 'minibatch' includes all the probability records, e.g.
    # images * cells * anchors...
    input_view = pred_class.view(-1, self.categories)
    target_view = ground_category_onehot.view(-1, self.categories)
    ce_loss_view = self.cross_entropy_loss(input_view, target_view)

    # Reshape the loss so we again have results for each image, cell, anchor...  (OLD YOLO COMMENT)
    #
    img_count,  _ = pred_class.shape
    classification_loss = ce_loss_view.view(img_count,-1)

    self.log_tensor(".classification_loss")
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
    TensorLogger.default_instance.report_grid(t, name, size=self.grid_size)
