#!/usr/bin/env python3

from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.printsize import *
import torch.nn.functional as F
from pycore.jg import JG

class JsModel(nn.Module):

  def __init__(self, network: NeuralNetwork):
    super(JsModel, self).__init__()

    # Issue #42: use a hard-coded network based on pytorch tutorial  ihttps://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    #
    if JG.HARD_CODED_NETWORK:
      self.conv1 = nn.Conv2d(3, # in channels
                             16, # out channels (# filters)
                             3, padding=1)  # kernel size
      self.pool1 = nn.MaxPool2d(2)   # input: 96x160, out: 48x80
      self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
      self.pool2 = nn.MaxPool2d(2)   # input: 48x80, out 24x40
      self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
      self.pool3 = nn.MaxPool2d(2)   # input: 24x40, out 16x20
      self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
      self.pool4 = nn.MaxPool2d(2)   # input: 12x20, out 6x10
      self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
      self.pool5 = nn.MaxPool2d(2)   # input: 6x10, out 3x5
      self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
      self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
      self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
      self.conv9 = nn.Conv2d(512, 256, 3, padding=1)
      self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
      self.conv11 = nn.Conv2d(512, 256, 3, padding=1)
      self.conv12 = nn.Conv2d(256, 512, 3, padding=1)
      self.fc1 = nn.Linear(512 * 3 * 5, 120)
      self.fc2 = nn.Linear(120, 3 * 5 * 6)
      return

    self.tensors = []
    self.add_size("image input")

    self.layer = None
    for lyr in network.layers:
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
    if JG.HARD_CODED_NETWORK:
      verify_not_nan("js_model_forward", x)
      x = self.conv1(x)
      verify_not_nan("conv1", x)
      x = F.relu(x)
      verify_not_nan("relu1",x)
      x = self.pool1(x)
      verify_not_nan("pool1", x)
      x = self.pool2(F.relu(self.conv2(x)))
      verify_not_nan("aaa", x)
      x = self.pool3(F.relu(self.conv3(x)))
      x = self.pool4(F.relu(self.conv4(x)))
      x = self.pool5(F.relu(self.conv5(x)))
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      x = F.relu(self.conv8(x))
      verify_not_nan("ggg", x)
      x = F.relu(self.conv9(x))
      x = F.relu(self.conv10(x))
      x = F.relu(self.conv11(x))
      x = F.relu(self.conv12(x))
      x = torch.flatten(x, 1)  # flatten all dimensions except batch
      x = F.relu(self.fc1(x))
      verify_not_nan("js_model_forward_final_relu", x)
      x = self.fc2(x)
      return x
    return self.layers(x)

