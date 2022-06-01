from numpy import ndarray

from pycore.base import *
import numpy as np
import torch
from torch import nn
from gen.tensor_info import *

class TensorLogger:


  def __init__(self, directory:str = "train_data"):
    self.number = 0
    self.dir = directory
    pass


  def add(self, tensor:torch.Tensor, name:str):
    ti = TensorInfo.new_builder()
    ti.name = name
    dt = tensor.dtype
    if dt == torch.float32:
      ti.data_type = DataType.FLOAT32
    elif dt == torch.int8:
      ti.data_type = DataType.UNSIGNED_BYTE
    elif dt == torch.int16:
      ti.data_type = DataType.UNSIGNED_SHORT
    else:
      die("unsupported data type:", dt)
    ti.shape = list(tensor.shape)
    self.write(ti, tensor)
    pr("added:",ti)


  def write(self, info:TensorInfo, tensor:torch.Tensor):
    self.number += 1
    p = self.get_path(".json")
    p_temp = self.temp_version(p)
    txt_write(p_temp, info.to_string(False))
    x = tensor.detach().numpy()
    b = self.get_path(".dat")
    b_temp = self.temp_version(b)
    x.tofile(b_temp)
    os.rename(p_temp, p)
    os.rename(b_temp, b)


  def clean(self, name:str):
    return name.replace(" ","_")


  def temp_version(self, name:str):
    result = name + ".tmp"
    if os.path.exists(result):
      die("path already exists:", result)
    return result


  def get_path(self, name:str):
    return os.path.join(self.dir, str(self.number) + self.clean(name))