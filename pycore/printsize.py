from torch import nn

from pycore.base import pr

class PrintSize(nn.Module):

  suppress = True #not warning("including layer printing")

  def __init__(self, message="<unknown>"):
    super(PrintSize, self).__init__()
    self.message = message

  def forward(self, x):
    if not PrintSize.suppress:
      if self.message == "!exit!":
        import sys
        sys.exit(0)
      elif self.message == "!stop!":
        PrintSize.suppress = True
      else:
        pr("Input shape:",f"'{self.message}'".ljust(16), list(x.shape))
    return x
