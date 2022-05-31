from gen.yolo import Yolo
from pycore.pytorch_util import *
from pycore.js_model import JsModel
from gen.neural_network import NeuralNetwork
from .yolo_util import *

# Derived from https://neptune.ai/blog/pytorch-loss-functions


class YoloLoss(nn.Module):

  def __init__(self, network: NeuralNetwork, yolo:Yolo):
    super(YoloLoss, self).__init__()
    self.network = network
    self.yolo = yolo
    self.num_anchors = anchor_box_count(yolo)

    # Construct a tensor containing the anchor boxes, normalized to the block size
    #

    # Default values, from  https://github.com/uvipen/Yolo-v2-pytorch/blob/master/src/yolo_net.py
    #   anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
    t = []
    bs = yolo.block_size
    b_x, b_y = 1.0 / bs.x, 1.0 / bs.y
    for abp in yolo.anchor_boxes_pixels:
      t.append((abp.x * b_x, abp.y * b_y))
    t = torch.Tensor(t)
    self.anchors = t


  # current: current output from the model being trained
  # target:  training labels we want it to converge to
  #
  def forward(self, current, target):
    y = self.yolo
    batch_size = current.data.size(0)
    gsize = grid_size(y)
    height = gsize.y
    width = gsize.x
    grid_cell_total = width * height


    # Reshape the target to match the current's shape?

    # Get x,y,w,h,conf,cls


    #  The -1 here makes it inferred from the other dimensions
    current = current.view(batch_size, self.num_anchors, -1, grid_cell_total)

    show("current", current)
    show("target",target)
    halt()


    todo("have ability to periodically send tensors to Java to store/report/inspect")

    # output shape: torch.Size([32, 1, 7, 169])
    #          32 = batch size
    #           1 = a single anchor box per grid cell
    #           7 = fields per anchor box               (inferred from other dimensions)
    #         169 = grid cells

    # Construct a tensor with room for just the box coords  (F_BOX_XYWH),
    # but with all other dimensions unchanged
    #
    coord = torch.zeros_like(current[:, :, F_BOX_XYWH:F_BOX_XYWH + 4, :])

    # Convert the predicted x,y (-inf...+inf) to 0...1 via sigmoid() function
    coord[:, :, F_BOX_X:F_BOX_Y+1, :] = current[:, :, F_BOX_X:F_BOX_Y + 1, :].sigmoid()

    # Convert predicted w,h (-inf...+inf) using exp function (not sure why)
    coord[:, :, F_BOX_W:F_BOX_H+1, :] = current[:, :, F_BOX_W:F_BOX_H + 1, :].exp()

    # Convert confidence (-inf...+inf) to probability 0...1 via sigmoid()
    #
    conf = current[:, :, F_CONFIDENCE, :].sigmoid()


    # For now, maybe we don't need to examine category probabilities?
    #
    cls = current[:, :, F_CLASS_PROBABILITIES:, :].contiguous().view(batch_size * self.num_anchors,
                                                                     y.category_count,
                                                                     grid_cell_total).transpose(1, 2).contiguous().view(-1,
                                                                                                  y.category_count)

    if FALSE:   # Shows that the above contiguous/transpose stuff is to reshape and rotate the tensor
      j = current[:, :, F_CLASS_PROBABILITIES:, :]
      show("j",j)
      show("cls",cls)
      halt()


    # Create prediction boxes
    pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * grid_cell_total, 4) # 4 is X,Y,W,H
    lin_x = torch.arange(0, width).repeat(height, 1).view(grid_cell_total)
    lin_y = torch.arange(0, height).repeat(width, 1).t().contiguous().view(grid_cell_total)

    if FALSE:
      # lin_x is an array of [0,1,2,...,W-1,  0,1,2,...,W-1, ...etc...]
      # lin_y is             [0,0,0, ..,0,    1,1,1,....1,   ...etc...]   ie lin_x is index mod W, lin_y is index/W
      show("lin_x",lin_x)
      show("lin_y",lin_y)

    anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
    anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

    if torch.cuda.is_available():
      pred_boxes = pred_boxes.cuda()
      lin_x = lin_x.cuda()
      lin_y = lin_y.cuda()
      anchor_w = anchor_w.cuda()
      anchor_h = anchor_h.cuda()

    # I think detach() is used to manipulate a Tensor that shouldn't participate in gradient calculations
    # and backpropagation...
    #
    #   "Tensor.detach()
    #    Returns a new Tensor, detached from the current graph.
    #
    #    The result will never require gradient."
    #
    pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)  # .view(-1) flattens it into an array
    pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
    pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
    pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
    pred_boxes = pred_boxes.cpu()   # I think this ensures it doesn't take up GPU space?

    # Get target values
    if FALSE:
      show("target", target)   # Tensor target torch.Size([32, 1183])
      halt()

    coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target, height, width)
    coord_mask = coord_mask.expand_as(tcoord)
    tcls = tcls[cls_mask].view(-1).long()
    cls_mask = cls_mask.view(-1, 1).repeat(1, y.category_count)

    if torch.cuda.is_available():
      tcoord = tcoord.cuda()
      tconf = tconf.cuda()
      coord_mask = coord_mask.cuda()
      conf_mask = conf_mask.cuda()
      tcls = tcls.cuda()
      cls_mask = cls_mask.cuda()

    conf_mask = conf_mask.sqrt()
    cls = cls[cls_mask].view(-1, y.category_count)

    # Compute losses
    mse = nn.MSELoss(size_average=False)
    ce = nn.CrossEntropyLoss(size_average=False)
    self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size
    self.loss_conf = mse(conf * conf_mask, tconf * conf_mask) / batch_size
    self.loss_cls = self.class_scale * 2 * ce(cls, tcls) / batch_size
    self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls

    return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_cls




  def build_targets(self, pred_boxes, ground_truth, height, width):
    pr("build_targets")
    show("pred_boxes",pred_boxes)
    show("ground_truth",ground_truth)

    y = self.yolo
    batch_size = len(ground_truth)
    grid_cell_total = height * width

    conf_mask  = torch.ones(batch_size,  self.num_anchors, grid_cell_total, requires_grad=False) * y.lambda_noobj
    coord_mask = torch.zeros(batch_size, self.num_anchors, 1, grid_cell_total, requires_grad=False)
    cls_mask   = torch.zeros(batch_size, self.num_anchors, grid_cell_total, requires_grad=False).byte()
    tcoord     = torch.zeros(batch_size, self.num_anchors, 4, grid_cell_total, requires_grad=False)
    tconf      = torch.zeros(batch_size, self.num_anchors, grid_cell_total, requires_grad=False)
    tcls       = torch.zeros(batch_size, self.num_anchors, grid_cell_total, requires_grad=False)

    image_label_size = len(ground_truth[0])

    for b in range(batch_size):

      # I think the ground truth was designed to be a sparse matrix, to only perform calculations on
      # grid cells / anchors that actually represent boxes
      #

      # # This looks suspect!
      # if len(ground_truth[b]) == 0:
      #   continue

      # Build up tensors
      x = (self.num_anchors * grid_cell_total)
      cur_pred_boxes = pred_boxes[b * x:(b + 1) * x]

      # stiches zeroes to the reference anchors, i.e. [0,0,  ax, ay]
      anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)


      gt = torch.zeros(image_label_size, 4)

      for i, anno in enumerate(ground_truth[b]):
        pr("i:",i,"b:",b,"anno:",anno)
        show("anno",anno)

        gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
        gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
        gt[i, 2] = anno[2] / self.reduction
        gt[i, 3] = anno[3] / self.reduction

      # Set confidence mask of matching detections to 0
      iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
      mask = (iou_gt_pred > self.thresh).sum(0) >= 1
      conf_mask[b][mask.view_as(conf_mask[b])] = 0

      # Find best anchor for each ground truth
      gt_wh = gt.clone()
      gt_wh[:, :2] = 0
      iou_gt_anchors = bbox_ious(gt_wh, anchors)
      _, best_anchors = iou_gt_anchors.max(1)

      # Set masks and target values for each ground truth
      for i, anno in enumerate(ground_truth[b]):
        gi = min(width - 1, max(0, int(gt[i, 0])))
        gj = min(height - 1, max(0, int(gt[i, 1])))
        best_n = best_anchors[i]
        iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
        coord_mask[b][best_n][0][gj * width + gi] = 1
        cls_mask[b][best_n][gj * width + gi] = 1
        conf_mask[b][best_n][gj * width + gi] = self.object_scale
        tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
        tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
        tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
        tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
        tconf[b][best_n][gj * width + gi] = iou
        tcls[b][best_n][gj * width + gi] = int(anno[4])

    return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls






def bbox_ious(boxes1, boxes2):
  b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
  b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
  b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
  b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

  dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
  dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
  intersections = dx * dy

  areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
  areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
  unions = (areas1 + areas2.t()) - intersections

  return intersections / unions
