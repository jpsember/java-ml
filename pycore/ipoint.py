from __future__ import annotations
from pycore.base import *

class IPoint(AbstractData):

  _key_x = "x"
  _key_y = "y"

  default_instance = None

  def __init__(self):
    self._x = 0
    self._y = 0

  @classmethod
  def new_builder(cls) -> IPointBuilder:
    return IPointBuilder()

  def parse(self, lst):
    if len(lst) != 2:
      die("can't parse IPoint from:", lst)
    return IPoint.with_x_y(lst[0],lst[1])

  def _x(self) -> int:
    return self._x

  def _y(self) -> int:
    return self._y

  x = property(_x)
  y = property(_y)


  @classmethod
  def with_x_y(cls, x, y):
    inst = IPoint()
    inst._x = x
    inst._y = y
    return inst


  def tuple(self):
    return self._x, self._y


  def to_json(self):
    return [self._x, self._y]


  def to_builder(self):
    x = IPointBuilder()
    x._x = self._x
    x._y = self._y
    return x

  def build(self):
    return self

  def product(self) -> int:
    return self._x * self._y

  def __hash__(self):
    return self._x + self._y

  def __eq__(self, other):
    if isinstance(other, IPoint):
      return self._x == other._x\
        and self._y == other._y
    else:
      return False


IPoint.default_instance = IPoint()


class IPointBuilder(IPoint):

  def set_x(self, x) -> IPointBuilder:
    self._x = x
    return self

  def set_y(self, x) -> IPointBuilder:
    self._y = x
    return self

  def to_builder(self):
    return self

  def build(self):
    return IPoint.with_x_y(self._x, self._y)


  x = property(IPoint._x, set_x)
  y = property(IPoint._y, set_y)
