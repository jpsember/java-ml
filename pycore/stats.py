from pycore.base import *

class Stats:

  def __init__(self, name:str):
    self.value = None
    self.value_sm = None
    self.name = name

  def set_value(self, value):
    self.value = value
    self.value_sm = smooth(self.value, self.value_sm)

  def info(self):
    s = self.name + ": "
    v = self.value
    if v is None:
      s += "(none)"
    else:
      s += f"{v:.5f} ({self.value_sm:.5f})"
    return s.strip()


def smooth(value, smoothed, t=0.05):
  return (none_to(smoothed, value) * (1 - t)) + (value * t)

