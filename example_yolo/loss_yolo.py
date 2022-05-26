from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork

# Derived from https://neptune.ai/blog/pytorch-loss-functions


class YoloLoss(nn.Module):


  def __init__(self, weight=None, size_average=True):
    super(YoloLoss, self).__init__()


  def forward(self, inputs, targets, smooth=1):
    warning("returning nonsensical loss value")
    return inputs.max()

    # inputs = F.sigmoid(inputs)
    #
    # inputs = inputs.view(-1)
    # targets = targets.view(-1)
    #
    # intersection = (inputs * targets).sum()
    # dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    #
    # return 1 - dice
