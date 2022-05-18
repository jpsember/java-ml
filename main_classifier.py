#!/usr/bin/env python3
from example_classifier.classifier_train import ClassifierTrain

c = ClassifierTrain()
c.prepare_pytorch()
c.run_training_session()
