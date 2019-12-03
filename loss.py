import torch
import torch.nn.functional as F
from math import exp
from cfg import Configs
import pytorch_ssim
import pytorch_iou


class MultiCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        loss_0 = F.cross_entropy(output[0], target[0],
                                 weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda()) / 64
        loss_1 = F.cross_entropy(output[1], target[1],
                                 weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda()) / 16
        loss_2 = F.cross_entropy(output[2], target[2],
                                 weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda()) / 4
        loss_3 = F.cross_entropy(output[3], target[3],
                                 weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda())
        total_loss = loss_0 + loss_1 + loss_2 + loss_3
        return total_loss

class HybridLoss(torch.nn.Module):
    def __init__(self, ce_weight, ssim_weight, iou_weight):
        super(HybridLoss, self).__init__()
        self.ce = MultiCrossEntropyLoss()
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
        self.iou_loss = pytorch_iou.IOU(size_average=True)
        self.ce_w = ce_weight
        self.ssim_w = ssim_weight
        self.iou_w = iou_weight

    def forward(self, output, target):
        ce_loss = self.ce(output, target)
        ssim_loss = 1- self.ssim_loss((torch.argmax(output[0],dim=1)).unsqueeze(1).float(), target[0].unsqueeze(1).float())
        iou_loss = self.iou_loss((torch.argmax(output[0],dim=1)).unsqueeze(1).float(), target[0].unsqueeze(1).float())
        return self.ce_w*ce_loss + self.ssim_w*ssim_loss + self.iou_w*iou_loss, ce_loss, ssim_loss, iou_loss


