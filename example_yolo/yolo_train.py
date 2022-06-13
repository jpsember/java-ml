#!/usr/bin/env python3
from gen.yolo import Yolo
from .yolo_model import YoloModel
from pycore.js_train import *
from .loss_yolo import YoloLoss


class YoloTrain(JsTrain):


  def __init__(self):
    super().__init__(__file__)
    JG.yolo = Yolo.default_instance.parse(self.network.model_config)
    #self.add_timeout(3600)


  def define_model(self):
    return YoloModel(self.network).to(self.device)


  def define_loss_function(self):
    return YoloLoss(self.network, self.logger, JG.yolo)
