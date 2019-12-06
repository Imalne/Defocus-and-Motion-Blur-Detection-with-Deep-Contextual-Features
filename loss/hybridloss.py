import torch
from loss.iou import IOU,MultiIOU
from loss.ssim import SSIM
from loss.multiCE import MultiCrossEntropyLoss
import torch.nn.functional as F

class HybridLoss(torch.nn.Module):
    def __init__(self, ce_weight, ssim_weight, iou_weight):
        super(HybridLoss, self).__init__()
        self.ce = MultiCrossEntropyLoss()
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.iou_loss = MultiIOU(size_average=True)
        self.ce_w = ce_weight
        self.ssim_w = ssim_weight
        self.iou_w = iou_weight

    def forward(self, output, target):
        ce_loss = self.ce(output, target)
        onehot_target = [torch.zeros(output[i].shape).cuda(output[i].get_device()).scatter_(1, target[i].unsqueeze(1), 1) for i in range(len(output))]
        output_prob = [F.softmax(i, dim=1) for i in output]
        iou_loss = self.iou_loss(output_prob, onehot_target)
        ssim_loss = self.ssim_loss(output_prob[0], onehot_target[0])
        return self.ce_w*ce_loss + self.ssim_w*ssim_loss + self.iou_w*iou_loss, ce_loss, ssim_loss, iou_loss