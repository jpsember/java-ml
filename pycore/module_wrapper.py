from pycore.pytorch_util import *
from pycore.tensor_logger import TensorLogger


# This is a wrapper for Modules that allows us to examine things for debug purposes,
# e.g. the input/output volumes
#
class ModuleWrapper(nn.Module):

  unique_id : int = 0

  def __init__(self):
    super(ModuleWrapper, self).__init__()
    self.show_size_flag = False
    self.message = None
    self.id = None


  def assign_id(self):
    self.id = self.unique_id
    ModuleWrapper.unique_id = self.id + 1
    return self


  def set_message(self, message):
    self.message = message
    return self


  def set_show_size_flag(self, flag=True):
    self.show_size_flag = flag
    return self


  def forward(self, tns):
    if self.show_size_flag:
      self.show_size_flag = False
      m = none_to(self.message,"<no message>")
      if self.id is not None:
        text = f"| {self.id:02} "
      else:
        text = "|    "
      text = text + spr("Input shape:",f"'{m}'".ljust(16), list(tns.shape))
      TensorLogger.default_instance.add_msg(text)
    return tns

