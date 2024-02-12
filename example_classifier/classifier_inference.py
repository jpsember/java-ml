#!/usr/bin/env python3

from .classifier_model import ClassifierModel

from pycore.js_inference import *


class ClassifierInference(JsInference):

  def __init__(self):
    super().__init__(__file__)


  def define_model(self):
    return ClassifierModel(self.network).to(self.device)

