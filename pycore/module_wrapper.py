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

    ep_list = self.log_input_vol_batch_list
    if ep_list is not None:
      if self.batch_number in ep_list:
        t = LogItem.new_builder()
        t.name = "input_vol"

        # Look only at a single plane of a single image's input volume
        #
        tens = tns
        tshp = tens.shape
        pr("shape of tensor:",tshp)
        imgnum = 0
        _, nfilt, h, w = tshp
        plane  = nfilt//2

        # Look at a centered subrect if the planar dimensions are larger than the desired subrect size
        #
        max_w = 12
        max_h = 8

        if h > max_h:
          y = (h - max_h)//2
          h = max_h
        if w > max_w:
          x = (w - max_w)//2
          w = max_w

        tens = tens[imgnum, plane, y:y+h, x:x+w]
        TensorLogger.default_instance.add(tens, t)

    self.batch_number += 1
    return tns


  def set_log_input_vol(self, batch_numbers=[5]):
    self.log_input_vol_batch_list = batch_numbers.copy()
    return self
