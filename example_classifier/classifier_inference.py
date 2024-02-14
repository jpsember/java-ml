#!/usr/bin/env python3

from .classifier_model import ClassifierModel

from pycore.js_inference import *


class ClassifierInference(JsInference):

  def __init__(self):
    super().__init__(__file__)
    self.classifier = None

  def define_model(self):
    self.classifier = getClassifier(self.network)
    return ClassifierModel(self.network, self.classifier).to(self.device)

