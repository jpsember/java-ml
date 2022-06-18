#!/usr/bin/env python3
from pycore.base import *

from example_yolo.yolo_train import YoloTrain

c = YoloTrain()
pr("preparing pytorch")
c.prepare_pytorch()
c.run_training_session()
