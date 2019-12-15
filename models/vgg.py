import torch
from torchvision.models.vgg import vgg19_bn,vgg19


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19,self).__init__()
        self.vgg = vgg19(pretrained=True)
        self.encoder_1 = self.vgg.features[0:4]
        self.encoder_2 = self.vgg.features[4:9]
        self.encoder_3 = self.vgg.features[9:18]
        self.encoder_4 = self.vgg.features[18:27]
        self.encoder_5 = self.vgg.features[27:36]

    def forward(self,x):
        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_1(skip_2)
        skip_4 = self.encoder_1(skip_3)
        x = self.encoder_5(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x


class VGG19_bn(torch.nn.Module):
    def __init__(self):
        super(VGG19_bn,self).__init__()
        self.vgg = vgg19_bn(pretrained=True)

        self.encoder_1 = self.vgg.features[0:6]
        self.encoder_2 = self.vgg.features[6:13]
        self.encoder_3 = self.vgg.features[13:26]
        self.encoder_4 = self.vgg.features[26:39]
        self.encoder_5 = self.vgg.features[39:-1]

    def forward(self,x):
        skip_1 = self.encoder_1(x)
        skip_2 = self.encoder_1(skip_1)
        skip_3 = self.encoder_1(skip_2)
        skip_4 = self.encoder_1(skip_3)
        x = self.encoder_5(skip_4)
        return skip_1, skip_2, skip_3, skip_4, x
