# Global variables for quick experiments
#
# because Python is such a pain in the ass
#

from __future__ import annotations
from gen.train_param import TrainParam

class JG:
  device = None
  train_param: TrainParam = None
  aux_stats: dict = None
  batch_number: int = None
