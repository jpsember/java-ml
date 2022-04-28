from pycore.base import AbstractData

class Matrix(AbstractData):

  _key_a = "a"
  _key_b = "b"
  _key_c = "c"
  _key_d = "d"
  _key_tx = "tx"
  _key_ty = "ty"

  def __init__(self):
    self._hash_value = None
    self._a = 1.0
    self._b = 0.0
    self._c = 0.0
    self._d = 1.0
    self._tx = 0.0
    self._ty = 0.0

  @classmethod
  def new_builder(cls):
    return MatrixBuilder()

  def parse(self, obj):
    x = Matrix()
    x._a, x._b, x._c, x._d, x.tx, x._ty = obj
    return x

  def _get_a(self):
    return self._a

  def _get_b(self):
    return self._b

  def _get_c(self):
    return self._c

  def _get_d(self):
    return self._d

  def _get_tx(self):
    return self._tx

  def _get_ty(self):
    return self._ty

  a = property(_get_a)
  b = property(_get_b)
  c = property(_get_c)
  d = property(_get_d)
  tx = property(_get_tx)
  ty = property(_get_ty)

  def to_builder(self):
    x = MatrixBuilder()
    x._a = self._a
    x._b = self._b
    x._c = self._c
    x._d = self._d
    x._tx = self._tx
    x._ty = self._ty
    return x

  def to_json(self):
    return [self._a,self._b,self._c,self._d,self._tx,self._ty]

  def __hash__(self):
    if self._hash_value is None:
      r = hash(self._a)
      r = r * 37 + hash(self._b)
      r = r * 37 + hash(self._c)
      r = r * 37 + hash(self._d)
      r = r * 37 + hash(self._tx)
      r = r * 37 + hash(self._ty)
      self._hash_value = r
    return self._hash_value

  def __eq__(self, other):
    if isinstance(other, Matrix):
      return hash(self) == hash(other)\
        and self._a == other._a\
        and self._b == other._b\
        and self._c == other._c\
        and self._d == other._d\
        and self._tx == other._tx\
        and self._ty == other._ty
    else:
      return False


Matrix.default_instance = Matrix()
Matrix.INDENTITY = Matrix.default_instance


class MatrixBuilder(Matrix):

  def set_a(self, x):
    self._a = 0.0 if x is None else x
    return self

  def set_b(self, x):
    self._b = 0.0 if x is None else x
    return self

  def set_c(self, x):
    self._c = 0.0 if x is None else x
    return self

  def set_d(self, x):
    self._d = 0.0 if x is None else x
    return self

  def set_tx(self, x):
    self._tx = 0.0 if x is None else x
    return self

  def set_ty(self, x):
    self._ty = 0.0 if x is None else x
    return self

  a = property(Matrix._get_a, set_a)
  b = property(Matrix._get_b, set_b)
  c = property(Matrix._get_c, set_c)
  d = property(Matrix._get_d, set_d)
  tx = property(Matrix._get_tx, set_tx)
  ty = property(Matrix._get_ty, set_ty)

  def to_builder(self):
    return self

  def build(self):
    v = Matrix()
    v._a = self._a
    v._b = self._b
    v._c = self._c
    v._d = self._d
    v._tx = self._tx
    v._ty = self._ty
    return v
