#!/usr/bin/env python3

# Driver for pytorch experiments
#


from pycore.base import *
from example_classifier.classifier_train import ClassifierTrain
from example_yolo.yolo_train import YoloTrain


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

  def select_project(self):
    self.project = next_arg_if("project", "yolo")


if __name__ == "__main__":
  try:
    App().run()
  except Exception as e:
    report_exception(e)
