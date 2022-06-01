from pycore.base import *
import numpy as np
import torch
from torch import nn
from gen.vol import *


def read_bytes(path: str, offset: int, record_size: int, record_count: int, convert_to_float:bool) -> np.ndarray:
  t = np.fromfile(path, dtype=np.int8, count=record_size * record_count, offset= offset)
  if convert_to_float:
    t = t.astype(np.float32) * (1.0 / 255) # This constant is the same as ImgUtil.RGB_TO_FLOAT
  t = t.reshape((record_count, record_size))
  return t


def read_floats(path: str, offset_in_floats: int, record_size_in_floats: int, record_count: int) -> np.ndarray:
  t = np.fromfile(path, dtype=np.float32, count=record_size_in_floats * record_count, offset= offset_in_floats * BYTES_PER_FLOAT)
  t = t.reshape((record_count, record_size_in_floats))
  return t


def read_ints(path: str, offset_in_ints: int, record_size_in_ints: int, record_count: int) -> np.ndarray:
  t = np.fromfile(path, dtype=np.int32, count=record_size_in_ints * record_count, offset= offset_in_ints * BYTES_PER_INT)
  t = t.reshape((record_count, record_size_in_ints))
  return t


def write_ints(path: str, ints: np.ndarray):
  """
  Writes a numpy array (of ints?) to a binary file, with little-endianness?
  """
  with open(path, "wb") as f:
    f.write(ints.tobytes())


def vol_volume(vol:Vol) -> int:
  return vol.width * vol.height * vol.depth


def show(label:str, obj):
  if label[0] == '.':
    return
  dash = "------------------------\n"
  pr("\n\n")
  pr(dash)
  if isinstance(obj, torch.Tensor):
    t:torch.Tensor = obj
    pr(label,"      size:",list(t.size()),"   type:",t.dtype)
    pr(t)
  else:
    pr(label,"type:",type(obj),"value:",obj)
  pr(dash)
  pr("\n\n")
