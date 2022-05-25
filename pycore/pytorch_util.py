from pycore.base import *
import numpy as np
import torch
from torch import nn
from gen.vol import *


def read_bytes(path: str, offset: int, record_size: int, record_count: int, convert_to_float:bool) -> np.ndarray:
  t = np.fromfile(path, dtype=np.int8, count=record_size * record_count, offset= offset)
  pr("read_bytes, record count:", record_count, "size:", record_size, "input shape:", t.shape)
  if convert_to_float:
    todo("can we just say t.byte()?")
    todo("scale from 0..255 to 0.0 ... 1.0")
    t = t.astype(np.float32)
  t = t.reshape((record_count, record_size))
  pr("...reshaped to:", t.shape)
  return t


def read_floats(path: str, offset_in_floats: int, record_size_in_floats: int, record_count: int) -> np.ndarray:
  t = np.fromfile(path, dtype=np.float32, count=record_size_in_floats * record_count, offset= offset_in_floats * BYTES_PER_FLOAT)
  pr("reshaping floats, record count:",record_count,"size in floats:",record_size_in_floats,"input shape:",t.shape)
  t = t.reshape((record_count, record_size_in_floats))
  pr("reshaped to:",t.shape)
  return t


def read_ints(path: str, offset_in_ints: int, record_size_in_ints: int, record_count: int) -> np.ndarray:
  t = np.fromfile(path, dtype=np.int32, count=record_size_in_ints * record_count, offset= offset_in_ints * BYTES_PER_INT)
  pr("read_ints, about to reshape from:",t.shape)
  t = t.reshape((record_count, record_size_in_ints))
  pr("reshaped to:",t.shape)
  return t


def write_ints(path: str, ints: np.ndarray):
  """
  Writes a numpy array (of ints?) to a binary file, with little-endianness?
  """
  with open(path, "wb") as f:
    f.write(ints.tobytes())


def vol_volume(vol:Vol) -> int:
  return vol.width * vol.height * vol.depth
