import torch
import torch.nn.functional as F

class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super(Decoder,self).__init__()

        self.deconv_1_1 = torch.nn.ConvTranspose2d(channels[0], channels[0], 3, 2, 1, 1)
        self.deconv_1_2 = torch.nn.Sequential(torch.nn.Conv2d(channels[0], channels[0], 3, 1, 1), torch.nn.BatchNorm2d(channels[0]), torch.nn.ReLU())
        self.deconv_1_3 = torch.nn.Sequential(torch.nn.Conv2d(channels[0], channels[0], 3, 1, 1), torch.nn.BatchNorm2d(channels[0]), torch.nn.ReLU())
        self.deconv_1_4 = torch.nn.Sequential(torch.nn.Conv2d(channels[0], channels[0], 3, 1, 1), torch.nn.BatchNorm2d(channels[0]), torch.nn.ReLU())

        self.deconv_2_1 = torch.nn.ConvTranspose2d(channels[0], channels[1], 3, 2, 1, 1)
        self.deconv_2_2 = torch.nn.Sequential(torch.nn.Conv2d(channels[1], channels[1], 3, 1, 1), torch.nn.BatchNorm2d(channels[1]), torch.nn.ReLU())
        self.deconv_2_3 = torch.nn.Sequential(torch.nn.Conv2d(channels[1], channels[1], 3, 1, 1), torch.nn.BatchNorm2d(channels[1]), torch.nn.ReLU())
        self.deconv_2_4 = torch.nn.Sequential(torch.nn.Conv2d(channels[1], channels[1], 3, 1, 1), torch.nn.BatchNorm2d(channels[1]), torch.nn.ReLU())

        self.deconv_3_1 = torch.nn.ConvTranspose2d(channels[1], channels[2], 3, 2, 1, 1)
        self.deconv_3_2 = torch.nn.Sequential(torch.nn.Conv2d(channels[2], channels[2], 3, 1, 1), torch.nn.BatchNorm2d(channels[2]), torch.nn.ReLU())
        self.deconv_3_3 = torch.nn.Sequential(torch.nn.Conv2d(channels[2], channels[2], 3, 1, 1), torch.nn.BatchNorm2d(channels[2]), torch.nn.ReLU())
        self.deconv_3_4 = torch.nn.Sequential(torch.nn.Conv2d(channels[2], channels[2], 3, 1, 1), torch.nn.BatchNorm2d(channels[2]), torch.nn.ReLU())

        self.deconv_4_1 = torch.nn.ConvTranspose2d(channels[2], channels[3], 3, 2, 1, 1)
        self.deconv_4_2 = torch.nn.Sequential(torch.nn.Conv2d(channels[3], channels[3], 3, 1, 1), torch.nn.BatchNorm2d(channels[3]), torch.nn.ReLU())
        self.deconv_4_3 = torch.nn.Sequential(torch.nn.Conv2d(channels[3], channels[3], 3, 1, 1), torch.nn.BatchNorm2d(channels[3]), torch.nn.ReLU())

        self.out_layer_1 = torch.nn.Conv2d(channels[3], 3, 1, 1, 0)
        self.out_layer_2 = torch.nn.Conv2d(channels[2], 3, 1, 1, 0)
        self.out_layer_3 = torch.nn.Conv2d(channels[1], 3, 1, 1, 0)
        self.out_layer_4 = torch.nn.Conv2d(channels[0], 3, 1, 1, 0)

    def forward(self, x, skip_1, skip_2, skip_3, skip_4):
        x = self.deconv_1_1(x)
        torch.add(x, skip_4)
        x = self.deconv_1_2(x)
        x = self.deconv_1_3(x)
        x = self.deconv_1_4(x)
        out_4 = self.out_layer_4(x)
        x = self.deconv_2_1(x)
        torch.add(x, skip_3)
        x = self.deconv_2_2(x)
        x = self.deconv_2_3(x)
        x = self.deconv_2_4(x)
        out_3 = self.out_layer_3(x)
        x = self.deconv_3_1(x)
        torch.add(x, skip_2)
        x = self.deconv_3_2(x)
        x = self.deconv_3_3(x)
        x = self.deconv_3_4(x)
        out_2 = self.out_layer_2(x)
        x = self.deconv_4_1(x)
        torch.add(x, skip_1)
        x = self.deconv_4_2(x)
        x = self.deconv_4_3(x)
        out_1 = self.out_layer_1(x)
        return F.softmax(out_1, dim=1), F.softmax(out_2, dim=1), F.softmax(out_3, dim=1), F.softmax(out_4, dim=1)