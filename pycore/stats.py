from pycore.base import *

class Stats:

  def __init__(self, name:str):
    self.value = None
    self.value_sm = None
    self.name = name


  def set_value(self, value):
    orig_sm = none_to(self.value_sm, value)
    self.value = value
    t = 0.05
    self.value_sm = (orig_sm * (1 - t)) + (value * t)


  def info(self, sig_digits = 3):
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
