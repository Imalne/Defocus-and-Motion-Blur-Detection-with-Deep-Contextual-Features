import torch

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19,   self).__init__()
        self.conv_1_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.ReLU())
        self.conv_1_2 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU())
        self.conv_1_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_2_1 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU())
        self.conv_2_2 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.ReLU())
        self.conv_2_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_3_1 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.ReLU())
        self.conv_3_2 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU())
        self.conv_3_3 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU())
        self.conv_3_4 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU())
        self.conv_3_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_4_1 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.ReLU())
        self.conv_4_2 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.conv_4_3 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.conv_4_4 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.conv_4_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_5_1 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.conv_5_2 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.conv_5_3 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.conv_5_4 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())

    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        skip_1 = x
        x = self.conv_1_mp(x)
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        skip_2 = x
        x = self.conv_2_mp(x)
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.conv_3_4(x)
        skip_3 = x
        x = self.conv_3_mp(x)
        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.conv_4_4(x)
        skip_4 = x
        x = self.conv_4_mp(x)
        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.conv_5_4(x)
        return skip_1, skip_2, skip_3, skip_4, x


class VGG19_bn(torch.nn.Module):
    def __init__(self):
        super(VGG19_bn,   self).__init__()
        self.conv_1_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.BatchNorm2d(64), torch.nn.ReLU())
        self.conv_1_2 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.BatchNorm2d(64), torch.nn.ReLU())
        self.conv_1_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_2_1 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.BatchNorm2d(128), torch.nn.ReLU())
        self.conv_2_2 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.BatchNorm2d(128), torch.nn.ReLU())
        self.conv_2_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_3_1 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU())
        self.conv_3_2 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU())
        self.conv_3_3 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU())
        self.conv_3_4 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU())
        self.conv_3_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_4_1 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
        self.conv_4_2 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
        self.conv_4_3 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
        self.conv_4_4 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
        self.conv_4_mp = torch.nn.MaxPool2d(2, 2, 0, 1, False)

        self.conv_5_1 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
        self.conv_5_2 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
        self.conv_5_3 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())
        self.conv_5_4 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU())

    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        skip_1 = x
        x = self.conv_1_mp(x)
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        skip_2 = x
        x = self.conv_2_mp(x)
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.conv_3_4(x)
        skip_3 = x
        x = self.conv_3_mp(x)
        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.conv_4_4(x)
        skip_4 = x
        x = self.conv_4_mp(x)
        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.conv_5_4(x)

        return skip_1, skip_2, skip_3, skip_4, x