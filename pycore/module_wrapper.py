from gen.log_item import LogItem
from pycore.pytorch_util import *
from pycore.tensor_logger import TensorLogger


class ModuleWrapper(nn.Module):

  unique_id : int = 0

  def __init__(self):
    super(ModuleWrapper, self).__init__()
    self.show_size_flag = False
    self.message = None
    self.id = None
    self.log_input_vol_epochs_list = None
    self.epoch_number = 0


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


  def forward(self, x):
    if self.show_size_flag:
      self.show_size_flag = False
      m = none_to(self.message,"<no message>")
      text = ""
      if self.id is not None:
        text = f"| {self.id:02} "
      else:
        text = "|    "
      text = text + spr("Input shape:",f"'{m}'".ljust(16), list(x.shape))
      TensorLogger.default_instance.add_msg(text)

    ep_list = self.log_input_vol_epochs_list
    if ep_list is not None:
      if self.epoch_number in ep_list:
        t = LogItem.new_builder()
        t.name = "input_vol"
        todo("assuming first dimension is image within batch")
        pr("shape:", x.shape)
        # Look only at first filter
        tens = x[1, 1, ...]
        TensorLogger.default_instance.add(tens, t)

    self.epoch_number += 1
    return x


  def set_log_input_vol(self, epochs=[5]):
    self.log_input_vol_epochs_list = epochs.copy()
    return self
