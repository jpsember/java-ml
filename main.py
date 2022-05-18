#!/usr/bin/env python3

from example_classifier.classifier_train import ClassifierTrain

print("Can we get rid of this file and start from the appropriate example subdir?")

c = ClassifierTrain()
c.prepare_pytorch()
c.run_training_session()
