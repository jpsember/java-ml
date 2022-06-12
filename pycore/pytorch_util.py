from pycore.base import *
import numpy as np
import torch
from torch import nn
from gen.vol import *
from pycore.ipoint import IPoint


def read_unsigned_bytes(path: str, offset: int, record_size: int, record_count: int) -> np.ndarray:
  t = np.fromfile(path, dtype=np.ubyte, count=record_size * record_count, offset= offset)
  t = t.reshape((record_count, record_size))
  return t


def convert_float_tensor_to_byte_tensor(t:torch.Tensor):
  return (t * (255 / 1.0)).to(torch.uint8)


def convert_unsigned_bytes_to_floats(t:np.ndarray):
  return t.astype(np.float32) * (1.0 / 255)  # This constant is the same as ImgUtil.RGB_TO_FLOAT


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



def pt_to_ftensor(pt:IPoint):
  return torch.FloatTensor(pt.tuple())


def get_var(var, name: str, depth: int = 1):
  if var is not None:
    return var

  # If name includes a suffix :xxx, omit it
  #
  suffix = name.find(':')
  if suffix >= 0:
    name = name[:suffix]
  import inspect
  frame = inspect.currentframe()
  for _ in range(1 + depth):
    frame = frame.f_back
  var = frame.f_locals[name]
  del frame
  return var


def show_shape(tensor_or_name, tensor=None):
  nm = tensor_or_name
  if tensor is None:
    if isinstance(tensor_or_name, torch.Tensor):
      tensor = tensor_or_name
      nm = "(unknown)"
    else:
      tensor = get_var(None, tensor_or_name, 1)
  pr(f"{nm:>20} s{list(tensor.shape)}")


def verify_not_nan(prompt, tensor_or_name, tensor=None):
  nm, tensor = get_name_and_tensor_pair(tensor_or_name, tensor)
  if not nm:
    return
  if torch.any(torch.isnan(tensor)):
    pr("Tensor",nm,"has NaN values")
    pr(tensor)
    die("(**************** NaN values found in", nm,":",prompt)


def verify_weights_not_nan(prompt, tensor_or_name, tensor=None):
  todo("refactor to reuse verify_not_nan with the weights?")
  nm, tensor = get_name_and_tensor_pair(tensor_or_name, tensor)
  if not nm:
    return
  wt = tensor.weight
  if torch.any(torch.isnan(wt)):
    pr("Tensor",nm,"has NaN values;",prompt)
    die("(**************** NaN values found in weights for layer", nm,";",prompt)


def verify_non_negative(tensor_or_name, tensor=None):
  nm, tensor = get_name_and_tensor_pair(tensor_or_name, tensor)
  if torch.min(tensor).data < 0:
    pr("Tensor",nm,"has negative values")
    pr(tensor)
    die("(**************** Negative values found in", nm)


def get_name_and_tensor_pair(tensor_or_name, tensor):
  nm = tensor_or_name
  if tensor is None:
    if isinstance(tensor_or_name, torch.Tensor):
      tensor = tensor_or_name
      nm = "(unknown)"
    else:
      tensor = get_var(None, tensor_or_name, 2)
  if nm.startswith("."):
    nm = None
  return nm, tensor