# Global variables for quick experiments
#
# because Python is such a pain in the ass
#

class JG:
  trainer = None
  HARD_CODED_NETWORK = True
  recent_images_input = None
  recent_labels_input = None
  # Until I put Yolo stuff back into its own subclasses, the yolo object will be here in the globals
  yolo = None
