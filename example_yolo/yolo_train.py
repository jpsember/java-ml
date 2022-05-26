#!/usr/bin/env python3

from .yolo_model import YoloModel
from pycore.js_train import *
from .loss_yolo import YoloLoss


class YoloTrain(JsTrain):


  def __init__(self):
    # Is this necessary?
    super().__init__(__file__)


  def define_model(self):
    return YoloModel(self.network).to(self.device)


  def define_loss_function(self):
    return YoloLoss()
