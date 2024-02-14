#!/usr/bin/env python3

# Derived from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
from gen.classifier import Classifier
from .classifier_model import ClassifierModel
from pycore.js_train import *
from .classifier_loss import ClassifierLoss


class ClassifierTrain(JsTrain):


  def __init__(self):
    super().__init__(__file__)
    self.classifier = None
    # With this model, we are interested in the classifier's accuracy, so we include
    # it in the test reports
    #
    todo("figure out how to report the accuracy")
    #self.stat_acc = Stats("Accuracy")


  def define_model(self):
    self.classifier = getClassifier(self.network)
    return ClassifierModel(self.network, self.classifier).to(self.device)


  def define_loss_function(self):
    return ClassifierLoss(self.network, self.classifier)


