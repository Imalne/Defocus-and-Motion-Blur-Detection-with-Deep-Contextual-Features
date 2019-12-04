import torch
from loss.iou import IOU
from loss.ssim import SSIM
from loss.multiCE import MultiCrossEntropyLoss
import torch.nn.functional as F

class HybridLoss(torch.nn.Module):
    def __init__(self, ce_weight, ssim_weight, iou_weight):
        super(HybridLoss, self).__init__()
        self.ce = MultiCrossEntropyLoss()
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.iou_loss = IOU(size_average=True)
        self.ce_w = ce_weight
        self.ssim_w = ssim_weight
        self.iou_w = iou_weight

    def forward(self, output, target):
        ce_loss = self.ce(output, target)
        onehot_target = torch.zeros(output[0].shape).cuda().scatter_(1, target[0].unsqueeze(1), 1)
        iou_loss = self.iou_loss(output[0], onehot_target)
        ssim_loss = 1 - self.ssim_loss(output[0], onehot_target)
        return self.ce_w*ce_loss + self.ssim_w*ssim_loss + self.iou_w*iou_loss, ce_loss, ssim_loss, iou_loss