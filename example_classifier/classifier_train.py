#!/usr/bin/env python3

# Derived from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

from .classifier_model import ClassifierModel
from pycore.js_train import *
from .classifier_loss import ClassifierLoss


class ClassifierTrain(JsTrain):


  def __init__(self):
    super().__init__(__file__)

    # With this model, we are interested in the classifier's accuracy, so we include
    # it in the test reports
    #
    todo("figure out how to report the accuracy")
    #self.stat_acc = Stats("Accuracy")


  def define_model(self):
    return ClassifierModel(self.network).to(self.device)


  def define_loss_function(self):

    todo("For flexibility, I think we need a custom 'classifier_loss' analogous to yolo_loss.py")
    return ClassifierLoss(self.network)


