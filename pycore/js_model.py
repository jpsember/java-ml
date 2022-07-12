#!/usr/bin/env python3
from gen.neural_network import NeuralNetwork
from pycore.module_wrapper import *
from pycore.jg import JG


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
    self.display_sizes = False


  # Called by JsTrain to prepare the model for use.
  # Code that used to be run in the constructor has been moved here to allow subclass's custom initialization code
  #
  def prepare(self):
    check_state(not self.prepared, "model is already prepared")
    self.prepared = True

    if JG.train_param.detect_anomalies:
      warning("Enabling 'detect_anomaly'")
      torch.autograd.set_detect_anomaly(True)

    self.tensors = []
    self.add_size("image input")

    self.layer = None
    for lyr in self.network.layers:
      self.layer = lyr

      if lyr.type == "conv":
        kernel_width = lyr.kernel_width
        our_stride = lyr.stride.tuple()
        t = nn.Conv2d(in_channels=lyr.input_volume.depth,
                      out_channels=lyr.output_volume.depth,
                      kernel_size=kernel_width,
                      stride=our_stride,
                      padding=kernel_width//2, # half padding
                      )
        self.add_layer(t)
        if lyr.batch_norm and not JG.train_param.disable_batch_norm:
          bn = nn.BatchNorm2d(lyr.output_volume.depth)
          self.add_layer(bn)
      elif lyr.type == "leaky_relu":
        self.add_layer(nn.LeakyReLU(0.1, inplace = True))
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

    self.layers = nn.Sequential(*self.tensors)



  def process_custom_layer(self, lyr):
    die("unsupported layer type:", lyr.type)


  def add_size(self, message) -> ModuleWrapper:
    w = ModuleWrapper()
    w.set_message(message)
    w.set_show_size_flag(self.display_sizes)
    self.tensors.append(w)
    return w


  def add_layer(self, layer, size_label=None):
    if layer is not None:
      self.add_size(none_to(size_label, self.layer.type)).assign_id()
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
