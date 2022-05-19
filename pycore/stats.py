from pycore.base import *

class Stats:

  def __init__(self):
    self.loss = None
    self.loss_sm = None
    self.accuracy = None
    self.accuracy_sm = None

  def set_loss(self, loss):
    self.loss = loss
    self.loss_sm = smooth(self.loss, self.loss_sm)

  def set_accuracy(self, acc):
    self.accuracy = acc
    self.accuracy_sm = smooth(self.accuracy, self.accuracy_sm)

  def info(self):
    if self.loss is None:
      return "(none)"
    s = ""
    if self.accuracy is not None:
      s += f" acc: {self.accuracy:.1f} ({self.accuracy_sm:.1f})"
    s += f" loss: {self.loss:.5f} ({self.loss_sm:.5f})"
    return s.strip()


def smooth(value, smoothed, t=0.05):
  return (none_to(smoothed, value) * (1 - t)) + (value * t)

