from pycore.js_train import smooth


class Stats:

  def __init__(self):
    self.loss = None
    self.loss_sm = None
    self.accuracy = None
    self.accuracy_sm = None

  def set_loss(self, loss):
    self.loss = loss
    self.loss_sm = smooth(self.loss, self.loss_sm)

  def set_accuracy(self, acc):
    self.accuracy = acc
    self.accuracy_sm = smooth(self.accuracy, self.accuracy_sm)

  def info(self):
    if self.loss is None:
      return "(none)"
    return f"acc:{self.accuracy:.1f}({self.accuracy_sm:.1f}) loss:{self.loss:.3f}({self.loss_sm:.3f})"
