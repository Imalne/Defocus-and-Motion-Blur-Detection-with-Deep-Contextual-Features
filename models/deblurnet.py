import torch
from models.decoder import Decoder
from models.vgg import VGG19_bn, VGG19
from models.resnet import ResNet152, ResNet34, ResNet50
import torch.optim as optim
import os

layer_channels = {
    "vgg19": [512, 256, 128, 64],
    "vgg19_bn": [512, 256, 128, 64],
    "resnet34": [512, 256, 128, 64],
    "resnet152": [2048, 1024, 512, 256],
    "resnet50": [2048, 1024, 512, 256],
}


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net,self).__init__()
        self.encoder_type = config["encoder_type"]
        self.decoder = Decoder(layer_channels[self.encoder_type])
        self.get_encoder()
        self.get_skip_layer()


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
            if self.load_pre_from_local():
                print("load pre-parameters:" + prep_path + '...')
                prep = torch.load(prep_path)
                model_dict = self.encoder.state_dict()
                prep = self.parameter_rename(prep, model_dict)
                pre_trained_dict = {k: v for k, v in prep.items() if k in model_dict}
                model_dict.update(pre_trained_dict)
                self.encoder.load_state_dict(model_dict)
                print("load complete")
            else:
                print("use pretrained model", self.encoder_type," from torchvision")
            return 0

    def save_model(self, ech, save_path):
        torch.save({
            'epoch': ech,
            'model_state_dict': self.state_dict(),
        }, save_path)

    def parameter_rename(self, org_dict, target_dict):
        if self.encoder_type == "vgg19" or self.encoder_type == "vgg19_bn":
            org_list = []
            target_list = []
            for k, _ in target_dict.items():
                if k.find("batches") < 0:
                    target_list.append(k)
            for k, _ in org_dict.items():
                if k.find("batches") < 0:
                    org_list.append(k)
            replace_index = range(len(target_list))
            for i in replace_index:
                org_dict[target_list[i]] = org_dict.pop(org_list[i])
            return org_dict
        elif self.encoder_type == "resnet152":
            return target_dict


    def get_encoder(self):
        if self.encoder_type == "vgg19":
            self.encoder = VGG19()
        elif self.encoder_type == "vgg19_bn":
            self.encoder = VGG19_bn()
        elif self.encoder_type == "resnet152":
            self.encoder = ResNet152()
        elif self.encoder_type == "resnet34":
            self.encoder = ResNet34()
        elif self.encoder_type == "resnet50":
            self.encoder = ResNet50()
        else:
            raise RuntimeError("invalid encoder type")

    def get_skip_layer(self):
        channel = layer_channels[self.encoder_type]
        if channel is not None:
            self.skip_1 = torch.nn.Conv2d(channel[3], channel[3], 1, 1, 0)
            self.skip_2 = torch.nn.Conv2d(channel[2], channel[2], 1, 1, 0)
            self.skip_3 = torch.nn.Conv2d(channel[1], channel[1], 1, 1, 0)
            self.skip_4 = torch.nn.Conv2d(channel[0], channel[0], 1, 1, 0)
        else:
            raise RuntimeError("invalid encoder type")

    def load_pre_from_local(self):
        return self.encoder_type == "vgg19" or self.encoder_type == "vgg19_bn"

    def optimizer_by_layer(self, encoder_lr, decoder_lr):
        params = [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.decoder.parameters(), "lr":decoder_lr},
            {"params": self.skip_1.parameters(), "lr": encoder_lr},
            {"params": self.skip_2.parameters(), "lr": encoder_lr},
            {"params": self.skip_3.parameters(), "lr": encoder_lr},
            {"params": self.skip_4.parameters(), "lr": encoder_lr}
        ]
        return optim.Adam(params=params, lr=encoder_lr)