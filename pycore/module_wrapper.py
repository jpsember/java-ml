from pycore.pytorch_util import *

class ModuleWrapper(nn.Module):

  def __init__(self):
    super(ModuleWrapper, self).__init__()
    self.show_size_flag = False
    self.message = None


  def set_message(self, message):
    self.message = message
    return self


  def set_show_size_flag(self, flag=True):
    self.show_size_flag = flag
    return self


  def forward(self, x):
    if self.show_size_flag:
      self.show_size_flag = False
      todo("Use logger to have Java print this")
      m = none_to(self.message,"<no message>")
      pr("Input shape:",f"'{m}'".ljust(16), list(x.shape))
    return x
