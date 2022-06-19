#!/usr/bin/env python3

from .pytorch_util import *

# A wrapper for a module that supports selective logging
#
class LoggerModule(nn.Module):

  def __init__(self, wrapped_module):
    super(LoggerModule, self).__init__()
    self.wrapped_module = wrapped_module


  def forward(self, x):
    pr("LoggerModule forward called, calling wrapper")
    x = self.wrapped_module(x)
    return x
