import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)

class MultiIOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(MultiIOU, self).__init__()
        self.size_average = size_average


    def forward(self, output, target):
        level_num = len(output)
        total_loss = 0
        for i in range(level_num):
            level_loss = _iou(output[i], target[i],self.size_average)/(4 ** (level_num - 1 - i))
            total_loss += level_loss
        # loss_0 = _iou(output[0], target[0],self.size_average)/64
        # loss_1 = _iou(output[1], target[1],self.size_average)/16
        # loss_2 = _iou(output[2], target[2],self.size_average)/4
        # loss_3 = _iou(output[3], target[3],self.size_average)

        # total_loss = loss_0 + loss_1 + loss_2 + loss_3
        return total_loss
