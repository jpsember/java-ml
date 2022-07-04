#!/usr/bin/env python3
from gen.yolo import Yolo
from .yolo_model import YoloModel
from pycore.js_train import *
from .yolo_loss import YoloLoss


class YoloTrain(JsTrain):


  def __init__(self):
    super().__init__(__file__)
    self.yolo = Yolo.default_instance.parse(self.network.model_config)
    #self.add_timeout(120)


  def define_model(self):
    return YoloModel(self.network, self.yolo).to(self.device)


  def define_loss_function(self):
    return YoloLoss(self.network, self.yolo)
