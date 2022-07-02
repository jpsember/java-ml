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
    self.log_input_vol_batch_list = None
    self.batch_number = 0


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

    ep_list = self.log_input_vol_batch_list
    if ep_list is not None:
      if self.batch_number in ep_list:
        t = LogItem.new_builder()
        t.name = "input_vol"
        # Look only at first image, and maybe first filter?
        tens = x
        tshp = tens.shape
        pr("shape of tensor:",tshp)
        imgnum = 0
        _, nfilt, h, w = tshp
        plane  = nfilt//2
        y = h//2
        x = w//2
        tens = tens[imgnum:imgnum+1, plane, y-4:y+4, x-4:x+4]
        TensorLogger.default_instance.add(tens, t)

    self.batch_number += 1
    return x


  def set_log_input_vol(self, batch_numbers=[5]):
    self.log_input_vol_batch_list = batch_numbers.copy()
    return self
