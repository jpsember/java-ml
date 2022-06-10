#!/usr/bin/env python3
from pycore.base import *

from example_yolo.yolo_train import YoloTrain

c = YoloTrain()
c.prepare_pytorch()
c.run_training_session()
