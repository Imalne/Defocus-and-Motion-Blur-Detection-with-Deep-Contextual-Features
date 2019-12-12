import torch
import torch.nn.functional as func
import numpy as np
from utils import get_encoder,get_fpn_skip


class FPN(torch.nn.Module):
    def __init__(self, encoder, skip_ins, skip_out):
        super(FPN, self).__init__()
        self.encoder = encoder

        self.skip_0 = torch.nn.Conv2d(skip_ins[0], skip_out, kernel_size=1)
        self.skip_1 = torch.nn.Conv2d(skip_ins[1], skip_out, kernel_size=1)
        self.skip_2 = torch.nn.Conv2d(skip_ins[2], skip_out, kernel_size=1)
        self.skip_3 = torch.nn.Conv2d(skip_ins[3], skip_out, kernel_size=1)
        self.skip_4 = torch.nn.Conv2d(skip_ins[4], skip_out, kernel_size=1)

        self.conv_4 = torch.nn.Conv2d(skip_out, skip_out, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(skip_out, skip_out, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(skip_out, skip_out, kernel_size=3, padding=1)
        self.conv_1 = torch.nn.Conv2d(skip_out, skip_out, kernel_size=3, padding=1)
        self.conv_0 = torch.nn.Conv2d(skip_out, skip_out, kernel_size=3, padding=1)

        self.smooth_1 = torch.nn.Conv2d(4 * skip_out, skip_out, kernel_size=3, padding=1)
        self.smooth_2 = torch.nn.Conv2d(skip_out, skip_out, kernel_size=3, padding=1)

        self.out_1 = torch.nn.Conv2d(skip_out,3,kernel_size=3, padding=1)
        self.out_2 = torch.nn.Conv2d(skip_out,3,kernel_size=3, padding=1)

    def forward(self, x):
        map_0,map_1,map_2,map_3,map_4 = self.encoder(x)

        skip_4 = self.conv_4(self.skip_4(map_4))
        skip_3 = self.conv_3(self.skip_3(map_3) + func.interpolate(input=skip_4, scale_factor=2, mode='nearest'))
        skip_2 = self.conv_2(self.skip_2(map_2) + func.interpolate(input=skip_3, scale_factor=2, mode='nearest'))
        skip_1 = self.conv_1(self.skip_1(map_1) + func.interpolate(input=skip_2, scale_factor=2, mode='nearest'))
        skip_0 = self.conv_0(self.skip_0(map_0) + func.interpolate(input=skip_1, scale_factor=2, mode='nearest'))

        upsample_4 = func.interpolate(input=skip_4, scale_factor=8, mode='nearest')
        upsample_3 = func.interpolate(input=skip_3, scale_factor=4, mode='nearest')
        upsample_2 = func.interpolate(input=skip_2, scale_factor=2, mode='nearest')
        upsample_1 = skip_1

        concat = torch.cat([upsample_4, upsample_3, upsample_2, upsample_1], dim=1)
        concat = self.smooth_1(concat)

        out = skip_0 + func.interpolate(input=concat, scale_factor=2, mode='nearest')

        out = self.smooth_2(out)

        return self.out_1(out), self.out_2(concat)

    @classmethod
    def fromConfig(cls,config):
        encoder = get_encoder(config)
        return FPN(encoder,get_fpn_skip(config),config['skip_out_channel'])

    def save_model(self, ech, save_path):
        torch.save({
            'epoch': ech,
            'model_state_dict': self.state_dict(),
        }, save_path)
