import torch
from cfg import Configs
import torch.nn.functional as F


class MultiCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        level_num = len(output)
        total_loss = 0
        for i in range(level_num):
            level_loss = F.cross_entropy(output[i], target[i],
                                 weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda(output[0].get_device()))/ (4 ** (level_num -1 - i))
            total_loss += level_loss
        # loss_1 = F.cross_entropy(output[1], target[1],
        #                          weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda(output[1].get_device())) / 16
        # loss_2 = F.cross_entropy(output[2], target[2],
        #                          weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda(output[2].get_device())) / 4
        # loss_3 = F.cross_entropy(output[3], target[3],
        #                          weight=torch.FloatTensor(Configs["cross_entropy_weights"]).cuda(output[3].get_device()))
        # total_loss = loss_0 + loss_1 + loss_2 + loss_3
        return total_loss