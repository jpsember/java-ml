#!/usr/bin/env python3

from .yolo_model import YoloModel
from pycore.js_train import *


class YoloTrain(JsTrain):


  def __init__(self):
    # Is this necessary?
    super().__init__(__file__)


  def define_model(self):
    return YoloModel(self.network).to(self.device)


  def labels_are_ints(self):
    return False


if __name__ == "__main__":
  c = YoloTrain()
  c.prepare_pytorch()
  c.run_training_session()
