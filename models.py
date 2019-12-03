import torch
from cfg import Configs
import os
import torch.nn.functional as F
from math import exp

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
        skip_1 = self.skip_1(x)
        x = self.conv_1_mp(x)
        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        skip_2 = self.skip_2(x)
        x = self.conv_2_mp(x)
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.conv_3_4(x)
        skip_3 = self.skip_3(x)
        x = self.conv_3_mp(x)
        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.conv_4_4(x)
        skip_4 = self.skip_4(x)
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

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.deconv_1_1 = torch.nn.ConvTranspose2d(512, 512, 3, 2, 1, 1)
        self.deconv_1_2 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.deconv_1_3 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())
        self.deconv_1_4 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU())

        self.deconv_2_1 = torch.nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.deconv_2_2 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU())
        self.deconv_2_3 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU())
        self.deconv_2_4 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU())

        self.deconv_3_1 = torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv_3_2 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.ReLU())
        self.deconv_3_3 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.ReLU())
        self.deconv_3_4 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.ReLU())

        self.deconv_4_1 = torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv_4_2 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU())
        self.deconv_4_3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU())

        self.out_layer_1 = torch.nn.Conv2d(64, 3, 1, 1, 0)
        self.out_layer_2 = torch.nn.Conv2d(128, 3, 1, 1, 0)
        self.out_layer_3 = torch.nn.Conv2d(256, 3, 1, 1, 0)
        self.out_layer_4 = torch.nn.Conv2d(512, 3, 1, 1, 0)

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
        return out_1, out_2, out_3, out_4

class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net,self).__init__()
        self.encoder = VGG19_bn()
        self.decoder = Decoder()
        self.encoder_type = config["encoder_type"]
        self.skip_1 = torch.nn.Conv2d(64, 64, 1, 1, 0)
        self.skip_2 = torch.nn.Conv2d(128, 128, 1, 1, 0)
        self.skip_3 = torch.nn.Conv2d(256, 256, 1, 1, 0)
        self.skip_4 = torch.nn.Conv2d(512, 512, 1, 1, 0)

    def forward(self, x):
        sk_1, sk_2, sk_3, sk_4, x = self.encoder(x)
        sk_1 = self.skip_1(sk_1)
        sk_2 = self.skip_2(sk_2)
        sk_3 = self.skip_3(sk_3)
        sk_4 = self.skip_4(sk_4)
        return self.decoder(x, sk_1, sk_2, sk_3, sk_4)

    def load_model(self, prep_path, save_path):
        if os.path.exists(save_path):
            print("load from saved model:" + save_path + '...')
            checkpoint = torch.load(save_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            ech = checkpoint['epoch']
            self.eval()
            print("load complete")
            return ech
        else:
            print("load pre-parameters:" + prep_path + '...')
            prep = torch.load(prep_path)
            model_dict = self.state_dict()
            prep = self.parameter_rename(prep, model_dict, range(32))
            pre_trained_dict = {k: v for k, v in prep.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            self.load_state_dict(model_dict)
            print("load complete")
            return 0

    def save_model(self, ech, save_path):
        torch.save({
            'epoch': ech,
            'model_state_dict': self.state_dict(),
        }, save_path)

    def parameter_rename(self, org_dict, target_dict, replace_index):
        if self.encoder_type == "vgg19" or self.encoder_type == "vgg19_bn":
            org_list = []
            target_list = []
            for k, _ in target_dict.items():
                if k.find("batches") < 0:
                    target_list.append(k)
            for k, _ in org_dict.items():
                if k.find("batches") < 0:
                    org_list.append(k)
            for i in replace_index:
                org_dict[target_list[i]] = org_dict.pop(org_list[i])
            return org_dict