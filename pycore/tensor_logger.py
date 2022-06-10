from numpy import ndarray

from pycore.base import *
import numpy as np
import torch
from torch import nn
from gen.log_item import *

class TensorLogger:


  def __init__(self, directory:str = "train_data"):
    self.dir = directory
    self.id = 0
    pass


  def add_msg(self, *args):
    ti = self.new_log_item()
    ti.message = spr(*args)
    self.write(ti)


  def new_log_item(self, log_item:LogItem = None) -> LogItemBuilder:
    if log_item is not None:
      ti = log_item.to_builder()
    else:
      ti = LogItem.new_builder()
    self.id += 1
    ti.id = self.id
    return ti


  def to_list(self, arg):
    if isinstance(arg, list):
      return arg
    return list(arg)


  def add(self, tensor:torch.Tensor, name_or_info):
    ti:LogItemBuilder
    if isinstance(name_or_info, str):
      nm:str = name_or_info
      ti = self.new_log_item()
      ti.set_message(nm)
    else:
      ti = self.new_log_item(name_or_info)
      self.store_tensor(ti, tensor)
    self.write(ti, tensor)


  def store_tensor(self, ti:LogItemBuilder, tensor:torch.Tensor):
    ti.shape = list(tensor.shape)
    ti.set_tensor_bytes(None).set_tensor_floats(None)
    dt = tensor.dtype
    if dt == torch.float32:
      x = torch.flatten(tensor)
      ti.set_tensor_floats(self.to_list(x.tolist()))
    elif dt == torch.uint8:
      x = torch.flatten(tensor)
      ti.set_tensor_bytes(self.to_list(x.tolist()))
    #elif dt == torch.int16:
      #ti.data_type = DataType.UNSIGNED_SHORT
    else:
      die("unsupported data type:", dt)



  def write(self, info:LogItem, tensor:torch.Tensor = None):
    if tensor is not None:
      info = info.to_builder()
      self.store_tensor(info, tensor)

    #check_state(info.id != 0,"no id in LogItem")
    p = self.get_path(info, ".json")
    p_temp = self.temp_version(p)

    #pr("writing log item id:",info.id,"to:",p)
    # This writes the byte and float arrays using standard json, which is quite inefficient, but who cares
    txt_write(p_temp, info.to_string(False))
    os.rename(p_temp, p)


  def clean(self, name:str):
    return name.replace(" ","_")


  def temp_version(self, name:str):
    result = name + ".tmp"
    if os.path.exists(result):
      die("path already exists:", result)
    return result


  def get_path(self, info:LogItem, name:str):
    return os.path.join(self.dir, f"{info.id:07d}{self.clean(name)}")
