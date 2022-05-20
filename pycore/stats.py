from pycore.base import *

class Stats:

  def __init__(self, name:str):
    self.value = None
    self.value_sm = None
    self.name = name

  def set_value(self, value):
    self.value = value
    self.value_sm = smooth(self.value, self.value_sm)

  def info(self, sig_digits = 2):
    s = self.name + ": "
    v = self.value
    if v is None:
      s += "(none)"
    elif sig_digits == 0:
      s += f"{int(v)} ({int(self.value_sm)})"
    else:
      f = "." + str(sig_digits) + "f"
      s += f"{v:{f}} ({self.value_sm:{f}})"
    return s.strip()


def smooth(value, smoothed, t=0.05):
  return (none_to(smoothed, value) * (1 - t)) + (value * t)

