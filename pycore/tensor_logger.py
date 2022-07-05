from __future__ import annotations
from numpy import ndarray

from pycore.pytorch_util import *
from pycore.base import *
from pycore.jg import JG
import numpy as np
import torch
from torch import nn
from gen.log_item import *

class TensorLogger:

  default_instance: TensorLogger = None

  def __init__(self, directory:str = "train_data"):
    self.dir = directory
    self.id = 0
    self.report_count = 0


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







  # Issue a report with of a tensor that has dimensions (Image, Row, Column)
  #
  # If maximum number of reports has already been issued, does nothing.
  # Otherwise, extracts the first image as a plane of values
  #
  def report_grid(self, name, size, depth=1):
    # Have we already issued the maximum number of reports?
    if self.report_count >= JG.train_param.max_log_count:
      return
    if name.startswith("."):
      return
    self.report_count += 1

    # If tensor not provided, assume name refers to a local variable in the caller's scope
    #
    t = get_var(None, name, depth+1)

    # Construct a slice of the tensor for inspection
    z = t.detach()

    height = size.y
    width = size.x

    z = z[0,:]
    z = z.view(height,width,-1)

    max_width = 20
    max_height = 16

    r0 = 0
    c0 = 0
    if width > max_width:
      c0 = (width - max_width) // 2
      width = max_width
    if height > max_height:
      r0 = (height - max_height) // 2
      height = max_height

    # Zoom in on the center grid cells
    #     ROWS COLS
    z = z[r0:r0+height, c0:c0+width, :]
    self.add(z, name)


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


TensorLogger.default_instance = TensorLogger()

