#!/usr/bin/env python3
from gen.yolo import Yolo
from .yolo_model import YoloModel
from pycore.js_train import *
from .loss_yolo import YoloLoss


class YoloTrain(JsTrain):


  def __init__(self):
    # Is this necessary?
    super().__init__(__file__)
    self.yolo = Yolo.default_instance.parse(self.network.model_config)


  def define_model(self):
    #
    # # Try parsing a Yolo object from the network
    # d = self.network.model_config
    # y = Yolo.default_instance.parse(d)
    # pr("parsed yolo:")
    # pr(y)
    #
    #
    return YoloModel(self.network).to(self.device)


  def define_loss_function(self):
    return YoloLoss(self.network, self.yolo)
