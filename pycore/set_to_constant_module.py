#!/usr/bin/env python3

from .pytorch_util import *


class SetToConstant(nn.Module):

  def __init__(self, const_value = 0.2):
    super(SetToConstant, self).__init__()
    self.constant_val = const_value


  def forward(self, current):
    x = current * 1e-8 + self.constant_val
    return x
