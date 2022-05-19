#!/usr/bin/env python3

# Derived from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

from .classifier_model import ClassifierModel
from pycore.js_train import *

class ClassifierTrain(JsTrain):

  def __init__(self):
    super().__init__(__file__)
    self.loss = None
    self.correct = None
    self.display_progress_counter = 10

  def define_model(self):
    return ClassifierModel(self.network).to(self.device)


  def init_test(self):
    self.loss = 0
    self.correct = 0


  def update_test(self, pred, tensor_labels):
    predicted_labels = pred.argmax(1)
    if self.display_progress_counter > 0:
      self.display_progress_counter -= 1
      pr("model prediction:")
      pr(pred)
      pr("predicted labels:")
      pr(predicted_labels)
      pr("truth labels:")
      pr(tensor_labels)
    self.loss += self.loss_fn(pred, tensor_labels).item()
    self.correct += (predicted_labels == tensor_labels).type(torch.float).sum().item()


  def finish_test(self,test_image_count: int):
    self.stat_test_acc.set_value((100.0 * self.correct) / test_image_count)
    self.stat_test.set_value(self.loss)


  def labels_are_ints(self):
    return True


if __name__ == "__main__":
  c = ClassifierTrain()
  c.prepare_pytorch()
  c.run_training_session()
