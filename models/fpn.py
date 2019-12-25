import torch
import torch.nn.functional as func
import numpy as np
from utils import get_encoder,get_fpn_skip


class FPN():
    @classmethod
    def fromConfig(cls, config):
        encoder = get_encoder(config)
        if config['fpn_out'] == 2:
            fpn = FPN_2(encoder, get_fpn_skip(config), config['skip_out_channel'])
            for param in fpn.parameters():
                param.requires_grad = True
        elif config['fpn_out'] == 3:
            fpn = FPN_3(encoder, get_fpn_skip(config), config['skip_out_channel'])
            for param in fpn.parameters():
                param.requires_grad = True
        return fpn


class FPN_2(torch.nn.Module):
    def __init__(self, encoder, skip_ins, skip_out):
        super(FPN_2, self).__init__()
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


class FPN_3(torch.nn.Module):
    def __init__(self, encoder, skip_ins, skip_out):
        super(FPN_3, self).__init__()
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

        self.smooth_0 = torch.nn.Conv2d(3 * skip_out, skip_out, kernel_size=3, padding=1)
        self.smooth_1 = torch.nn.Conv2d(4 * skip_out, skip_out, kernel_size=3, padding=1)
        self.smooth_2 = torch.nn.Conv2d(skip_out, skip_out, kernel_size=3, padding=1)

        self.out_0 = torch.nn.Conv2d(skip_out, 3, kernel_size=3, padding=1)
        self.out_1 = torch.nn.Conv2d(skip_out,3,kernel_size=3, padding=1)
        self.out_2 = torch.nn.Conv2d(skip_out,3,kernel_size=3, padding=1)

    def forward(self, x):
        map_0, map_1, map_2, map_3, map_4 = self.encoder(x)

        skip_4 = self.conv_4(self.skip_4(map_4))
        skip_3 = self.conv_3(self.skip_3(map_3) + func.interpolate(input=skip_4, scale_factor=2, mode='nearest'))
        skip_2 = self.conv_2(self.skip_2(map_2) + func.interpolate(input=skip_3, scale_factor=2, mode='nearest'))
        skip_1 = self.conv_1(self.skip_1(map_1) + func.interpolate(input=skip_2, scale_factor=2, mode='nearest'))
        skip_0 = self.conv_0(self.skip_0(map_0) + func.interpolate(input=skip_1, scale_factor=2, mode='nearest'))

        # out_0
        upsample_0_4 = func.interpolate(input=skip_4, scale_factor=4, mode='nearest')
        upsample_0_3 = func.interpolate(input=skip_3, scale_factor=2, mode='nearest')
        upsample_0_2 = skip_2

        concat_0 = torch.cat([upsample_0_4,upsample_0_3,upsample_0_2], dim=1)
        concat_0 = self.smooth_0(concat_0)

        # out_1
        upsample_1_4 = func.interpolate(input=skip_4, scale_factor=8, mode='nearest')
        upsample_1_3 = func.interpolate(input=skip_3, scale_factor=4, mode='nearest')
        upsample_1_2 = func.interpolate(input=skip_2, scale_factor=2, mode='nearest')
        upsample_1_1 = skip_1

        concat_1 = torch.cat([upsample_1_4, upsample_1_3, upsample_1_2, upsample_1_1], dim=1)
        concat_1 = self.smooth_1(concat_1)

        # out_2
        out = skip_0 + func.interpolate(input=concat_1, scale_factor=2, mode='nearest')

        out = self.smooth_2(out)

        return self.out_2(out), self.out_1(concat_1), self.out_0(concat_0)

