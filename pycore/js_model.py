#!/usr/bin/env python3

from pycore.pytorch_util import *
from gen.neural_network import NeuralNetwork
from pycore.printsize import *
import torch.nn.functional as F
from pycore.jg import JG
from example_yolo.yolo_util import *

class JsModel(nn.Module):

  def __init__(self, network: NeuralNetwork):
    super(JsModel, self).__init__()
    self.debug_forward_counter = 0

    # Issue #42: use a hard-coded network based on pytorch tutorial  ihttps://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    #
    self.conv1 = nn.Conv2d(in_channels=3, # in channels
                           out_channels=16, # out channels (# filters)
                           kernel_size=3,
                           padding=1)  # kernel size
    self.pool1 = nn.MaxPool2d(2)   # input: 96x160, out: 48x80
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool2 = nn.MaxPool2d(2)   # input: 48x80, out 24x40
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool3 = nn.MaxPool2d(2)   # input: 24x40, out 16x20
    self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool4 = nn.MaxPool2d(2)   # input: 12x20, out 6x10
    self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
    self.pool5 = nn.MaxPool2d(2)   # input: 6x10, out 3x5
    self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
    self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
    self.conv9 = nn.Conv2d(512, 256, 3, padding=1)
    self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
    self.conv11 = nn.Conv2d(512, 256, 3, padding=1)
    self.conv12 = nn.Conv2d(256, 512, 3, padding=1)
    self.fc1 = nn.Linear(512 * 3 * 5, 120)
    self.fc2 = nn.Linear(120, 3 * 5 * 6)

    warning("Enabling 'detect_anomaly'")
    torch.autograd.set_detect_anomaly(True)

    self.batch_size = None
    self.num_anchors = None
    self.grid_size = None
    self.grid_cell_total = None
    # self.yolo = None
    # pr("set self.yolo to None in js_model")


  def verify_weights(self, message):
    verify_weights_not_nan(message, "conv1", self.conv1)


  def forward(self, x):
    verify_not_nan("js_model_forward", x)
    if self.debug_forward_counter == 1:
      self.verify_weights("before applying conv1")
    x = self.conv1(x)
    verify_not_nan("conv1", x)
    x = F.relu(x)
    verify_not_nan("relu1",x)
    x = self.pool1(x)
    verify_not_nan("pool1", x)
    x = self.pool2(F.relu(self.conv2(x)))
    verify_not_nan("aaa", x)
    x = self.pool3(F.relu(self.conv3(x)))
    x = self.pool4(F.relu(self.conv4(x)))
    x = self.pool5(F.relu(self.conv5(x)))
    x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))
    x = F.relu(self.conv8(x))
    verify_not_nan("ggg", x)
    x = F.relu(self.conv9(x))
    x = F.relu(self.conv10(x))
    x = F.relu(self.conv11(x))
    x = F.relu(self.conv12(x))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    verify_not_nan("js_model_forward_final_relu", x)
    x = self.fc2(x)

    # Apply narrowing functions to appropriate fields now, so we don't need to do them in the loss function or the
    # Java code
    # Each of these dimensions corresponds to (D_IMAGE, D_GRIDCELL, ..., D_BOXINFO)
    #
    y = JG.yolo

    if self.batch_size is None:
      self.batch_size = x.data.size(0)
      self.num_anchors = anchor_box_count(y)
      self.grid_size = grid_size(y)
      self.grid_cell_total = self.grid_size.product()








    current = x.view(self.batch_size, self.grid_cell_total, self.num_anchors, -1)  # -1 : infer remaining

    # ground_cxcy = target[:, :, :, F_BOX_CX:F_BOX_CY + 1]
    # self.log_tensor(".ground_cxcy")
    #
    # # true_box_wh will be the width and height of the box, relative to the anchor box
    # #
    # ground_wh = target[:, :, :, F_BOX_W:F_BOX_H+1]
    # self.log_tensor(".ground_wh")
    #
    #
    # # Determine number of ground truth boxes.  Clamp to minimum of 1 to avoid divide by zero.
    # #
    # true_box_count = torch.clamp(torch.count_nonzero(true_confidence), min=1).to(torch.float)
    #
    # # Add a dimension to true_confidence so it has equivalent dimensionality as ground_cxcy, ground_wh, etc
    # # (this doesn't change its volume, only its dimensions) <-- explain?
    # #
    # # This produces a mask value which we apply to the xy and wh loss.
    # coord_mask = true_confidence[..., None]

    # We could have stored the true class number as an index, instead of a one-hot vector;
    # but the symmetry of the structure of the true vs inferred data keeps things simple.
    #
    class_prob_end = F_CLASS_PROBABILITIES + y.category_count

    # Determine predicted box's x,y
    #
    # We need to map (-inf...+inf) to (0...1); hence apply sigmoid function
    #
    pred_cxcy = torch.sigmoid(current[:, :, :, F_BOX_CX:F_BOX_CY + 1])

    # Determine each predicted box's w,h
    #
    # We need to map (-inf...+inf) to (0..+inf); hence apply the exp function
    #
    pred_wh = torch.exp(current[:, :, :, F_BOX_W:F_BOX_H+1])

    # Determine each predicted box's confidence score.
    # We need to map (-inf...+inf) to (0..1); hence apply sigmoid function
    #
    pred_objectness = torch.sigmoid(current[:, :, :, F_CONFIDENCE:F_CONFIDENCE+1])

    # TODO: apply narrowing to the categories
    pred_categories = current[:, :, :, F_CLASS_PROBABILITIES:class_prob_end]

    x = torch.cat((pred_cxcy, pred_wh, pred_objectness, pred_categories), D_BOXINFO)

    self.debug_forward_counter += 1
    return x

