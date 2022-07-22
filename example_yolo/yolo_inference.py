#!/usr/bin/env python3
from gen.yolo import Yolo
from pycore.js_inference import *
from .yolo_model import YoloModel


class YoloInference(JsInference):

  def __init__(self):
    super().__init__(__file__)
    self.yolo = Yolo.default_instance.parse(self.network.model_config)


  def define_model(self):
    return YoloModel(self.network, self.yolo).to(self.device)

