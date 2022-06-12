#!/usr/bin/env python3

from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.printsize import *
import torch.nn.functional as F
from pycore.jg import JG

class JsModel(nn.Module):

  def __init__(self, network: NeuralNetwork):
    super(JsModel, self).__init__()
    self.debug_forward_counter = 0

    # Issue #42: use a hard-coded network based on pytorch tutorial  ihttps://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    #
    self.conv1 = nn.Conv2d(in_channels=3, # in channels
                           out_channels=16, # out channels (# filters)
                           kernel_size=3,
                           padding=1)  # kernel size
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

    warning("Enabling 'detect_anomaly'")
    torch.autograd.set_detect_anomaly(True)


  def verify_weights(self, message):
    verify_weights_not_nan(message, "conv1", self.conv1)


  def forward(self, x):
    pr("forward,", self.debug_forward_counter)
    verify_not_nan("js_model_forward", x)
    if self.debug_forward_counter == 1:
      self.verify_weights("before applying conv1")
    warning("applying conv1 is producing NaN from reasonable inputs between 0...1")
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
    pr("...done forward")
    self.debug_forward_counter += 1
    return x

