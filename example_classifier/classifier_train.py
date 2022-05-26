#!/usr/bin/env python3

# Derived from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

from .classifier_model import ClassifierModel
from pycore.js_train import *


class ClassifierTrain(JsTrain):


  def __init__(self):
    super().__init__(__file__)

    # With this model, we are interested in the classifier's accuracy, so we include
    # it in the test reports
    #
    self.stat_acc = Stats("Accuracy")


  def define_model(self):
    return ClassifierModel(self.network).to(self.device)


  def define_loss_function(self):
    return nn.CrossEntropyLoss()


  def update_test(self, pred, tensor_labels, test_image_count:int):
    predicted_labels = pred.argmax(1)
    if self.show_test_labels():
      pr("model prediction:")
      pr(pred)
      pr("predicted labels:")
      pr(predicted_labels)
      pr("truth labels:")
      pr(tensor_labels)
    correct = (predicted_labels == tensor_labels).type(torch.float).sum().item()
    self.stat_acc.set_value((100.0 * correct) / test_image_count)


  def test_target_reached(self) -> bool:
    return self.stat_acc.value_sm >= self.train_config.target_accuracy


  # Append the accuracy to the test report
  #
  def test_report(self) -> str:
    return super().test_report() + f"  {self.stat_acc.info(0)}"
