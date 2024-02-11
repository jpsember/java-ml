#!/usr/bin/env python3

# Driver for pytorch experiments
#

### I needed to run this command on my mac:
###  pip3 install jstyleson


from pycore.base import *
from example_classifier.classifier_train import ClassifierTrain
from example_yolo.yolo_train import YoloTrain
from example_yolo.yolo_inference import YoloInference


class App:

  def __init__(self):
    self.project = None


  def run(self):
    ca_builder(self)
    execute_commands()

  #-------------------------------------------------------------------------------------

  def help_train(self):
    return "[project <name>]"

  def perform_train(self):
    while handling_args():
      self.select_project()

    if self.project == "yolo":
      c = YoloTrain()
    elif self.project == "classifier":
      c = ClassifierTrain()
    else:
      die("Project not supported:", self.project)

    c.prepare_pytorch()
    c.run_training_session()

  #-------------------------------------------------------------------------------------

  def help_inference(self):
    return "[project <name>]"

  def perform_inference(self):
    while handling_args():
      self.select_project()

    if self.project == "yolo":
      c = YoloInference()
    else:
      die("Project not supported for 'perform_inference':", self.project)

    c.prepare_pytorch()
    c.run_inference()


  #-------------------------------------------------------------------------------------

  def select_project(self):
    self.project = next_arg_if("project", "yolo")


if __name__ == "__main__":
  try:
    App().run()
  except Exception as e:
    report_exception(e)
