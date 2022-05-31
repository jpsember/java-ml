#!/usr/bin/env python3


# Run this within a Python interactive shell by typing:
#
# python3
# import exp; import importlib
# importlib.reload(exp); exp.g()
#
# 1) Edit the file within PyCharm
# 2) Use the up arrow + return to reload/run the modified script
#


from pycore.pytorch_util import *

def db(label:str, obj):
  t: torch.Tensor = obj

  pr("\n\n\n")
  dash = "==========================================="
  pr(dash);
  pr(label,"size:",t.size())
  pr("\n")
  pr(t.numpy())
  pr(dash);

def g():
  pr("\n\n\n\n\n\n\n\n\n\n")
  t = torch.FloatTensor([[1.2,-1.2,3.0],[0.2,0.3,-0.3]])
  db("t",t)
  db("exp",t.exp())

