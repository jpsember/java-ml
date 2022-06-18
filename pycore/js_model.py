#!/usr/bin/env python3

from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.printsize import *
import torch.nn.functional as F
from pycore.jg import JG
from example_yolo.yolo_util import *


# Enapsulates the a Neural Network model
#
class JsModel(nn.Module):

  def __init__(self, network: NeuralNetwork):
    super(JsModel, self).__init__()
    self.network = network
    self.prepared = False
    self.tensors = None
    self.layers = None
    self.layer = None


  # Called by JsTrain to prepare the model for use.
  # Code that used to be run in the constructor has been moved here to allow subclass's custom initialization code
  #
  def prepare(self):
    check_state(not self.prepared, "model is already prepared")
    self.prepared = True

    warning("Enabling 'detect_anomaly'")
    torch.autograd.set_detect_anomaly(True)

    self.tensors = []
    self.add_size("image input")

    self.layer = None
    for lyr in self.network.layers:
      self.layer = lyr

      if lyr.type == "conv":
        our_stride = lyr.stride.tuple()
        t = nn.Conv2d(in_channels=lyr.input_volume.depth,
                      out_channels=lyr.output_volume.depth,
                      kernel_size=lyr.kernel_width,
                      stride=our_stride,
                      padding=lyr.kernel_width//2, # half padding
                      )
        self.add_layer(t)
      elif lyr.type == "leaky_relu":
        self.add_layer(nn.LeakyReLU())
      elif lyr.type == "maxpool":
        # I am used to thinking of a maxpool has having a stride, but they expect a kernel_size parameter
        stride = 2  # Ignoring any stride parameter in the layer!
        self.add_layer(nn.MaxPool2d(kernel_size=stride))
      elif lyr.type == "fc":
        self.add_layer(self.construct_fc())
      elif lyr.type == "output":
        self.construct_output()
      else:
        self.process_custom_layer(lyr)
    self.layer = None

    self.stop_size_messages()
    self.layers = nn.Sequential(*self.tensors)




  def process_custom_layer(self, lyr):
    die("unsupported layer type:", lyr.type)


  def add_size(self, message):
    self.tensors.append(PrintSize(message))


  def stop_size_messages(self):
    self.add_size("!stop!")


  def exit_when_built(self):
    self.add_size("!exit!")


  def add_layer(self, layer, size_label=None):
    if layer is not None:
      self.add_size(none_to(size_label, self.layer.type))
      self.tensors.append(layer)


  def last_tensor(self):
    return self.tensors[-1]


  def construct_fc(self):
    # We need to 'convert' the input volume into a set of confidences, one for each category.
    #
    # Flatten the input volume into a fibre, then apply a linear layer?
    #
    # See: https://stackoverflow.com/a/60372416
    #
    # Flatten:  https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    #
    ly = self.layer
    self.add_layer(nn.Flatten(),"fc.Flatten")
    self.add_layer(nn.Linear(vol_volume(ly.input_volume), ly.filters),"fc.Linear")


  def construct_output(self):
    # Reshape to fibre, if necessary
    #
    in_vol = self.layer.input_volume
    if in_vol.width != 1 or in_vol.height != 1:
      self.add_layer(nn.Flatten(), "output.Flatten")


  def forward(self, x):
    return self.layers(x)

