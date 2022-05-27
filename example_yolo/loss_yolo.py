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

    # TODO: let's get rid of these instance fields and just call self.yolo.xxxx
    #
    self.num_classes = yolo.category_count
    self.num_anchors = anchor_box_count(yolo)
    #pr("yolo:", yolo)
    pr("num_classes:",self.num_classes)
    pr("num_anchors:",self.num_anchors)


    # From  https://github.com/uvipen/Yolo-v2-pytorch/blob/master/src/yolo_net.py
    #
    #   anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
    #                           (11.2364, 10.0071)])
    #
    a = []
    bs = yolo.block_size
    for abp in yolo.anchor_boxes_pixels:
      a.append((abp.x / float(bs.x),abp.y / float(bs.y)))
    pr("anchors:",type(a),a)
    a = torch.Tensor(a)
    pr("anchors:",type(a),a)
    self.anchors = a


  def forward(self, output, target):
    y = self.yolo
    pr("forward, output:", output)
    pr("target:",target)
    pr("output.data.shape:",output.data.shape)

    warning("I think my layer is a single array of length n, and the code I am converting expects it to be 2 or more dimensions")

    batch_size = output.data.size(0)
    labels_size = output.data.size(1)
    gsize = grid_size(y)
    pr("batch_size:",batch_size)
    pr("labels_size:",labels_size)
    pr("values_per_block:",values_per_block(y))
    pr("float_labels:",image_label_float_count(y))
    pr("grid_size:",gsize)

    height = gsize.y
    width = gsize.x


    # Get x,y,w,h,conf,cls
    output = output.view(batch_size, self.num_anchors, -1, height * width)

    pr("output shape:",output.shape)
    # output shape: torch.Size([32, 1, 7, 169])
    #          32 = batch size
    #           1 = a single anchor box per grid cell
    #           7 = fields per anchor box
    #         169 = grid cells

    # Construct a tensor containing just the coordinates of the box  (F_BOX_XYWH)
    #
    coord = torch.zeros_like(output[:, :, :4, :])

    # Convert the predicted x,y (-inf...+inf) to 0...1 via sigmoid() function
    coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()

    # Convert predicted w,h (-inf...+inf) to ...?  **NOTE: Java code applies e^n here (exp function)
    coord[:, :, 2:4, :] = output[:, :, 2:4, :]

    # Convert confidence (-inf...+inf) to probability 0...1 via sigmoid()
    #
    conf = output[:, :, 4, :].sigmoid()

    # For now, maybe we don't need to examine category probabilities?
    #
    cls = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
                                                height * width).transpose(1, 2).contiguous().view(-1,
                                                                                                  self.num_classes)

    # Create prediction boxes
    pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
    lin_x = torch.range(0, width - 1).repeat(height, 1).view(height * width)
    lin_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(height * width)

    todo("IDE complaining about 'contiguous'")
    anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
    anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

    if torch.cuda.is_available():
      pred_boxes = pred_boxes.cuda()
      lin_x = lin_x.cuda()
      lin_y = lin_y.cuda()
      anchor_w = anchor_w.cuda()
      anchor_h = anchor_h.cuda()

    pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
    pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
    pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
    pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
    pred_boxes = pred_boxes.cpu()

    # Get target values
    coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target, height, width)
    coord_mask = coord_mask.expand_as(tcoord)
    tcls = tcls[cls_mask].view(-1).long()
    cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)

    if torch.cuda.is_available():
      tcoord = tcoord.cuda()
      tconf = tconf.cuda()
      coord_mask = coord_mask.cuda()
      conf_mask = conf_mask.cuda()
      tcls = tcls.cuda()
      cls_mask = cls_mask.cuda()

    conf_mask = conf_mask.sqrt()
    cls = cls[cls_mask].view(-1, self.num_classes)

    # Compute losses
    mse = nn.MSELoss(size_average=False)
    ce = nn.CrossEntropyLoss(size_average=False)
    self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size
    self.loss_conf = mse(conf * conf_mask, tconf * conf_mask) / batch_size
    self.loss_cls = self.class_scale * 2 * ce(cls, tcls) / batch_size
    self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls

    return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_cls

  def build_targets(self, pred_boxes, ground_truth, height, width):
    batch_size = len(ground_truth)

    conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale
    coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False)
    cls_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).byte()
    tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
    tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
    tcls = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)

    for b in range(batch_size):
      if len(ground_truth[b]) == 0:
        continue

      # Build up tensors
      cur_pred_boxes = pred_boxes[
                       b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]
      if self.anchor_step == 4:
        anchors = self.anchors.clone()
        anchors[:, :2] = 0
      else:
        anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
      gt = torch.zeros(len(ground_truth[b]), 4)
      for i, anno in enumerate(ground_truth[b]):
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