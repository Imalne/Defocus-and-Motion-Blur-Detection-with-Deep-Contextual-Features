from models.vgg import VGG19_bn, VGG19
from models.resnet import ResNet152, ResNet34, ResNet50

layer_channels = {
    "vgg19": [512, 512, 256, 128, 64],
    "vgg19_bn": [512, 512, 256, 128, 64],
    "resnet34": [512, 512, 256, 128, 64],
    "resnet152": [2048, 2048, 1024, 512, 256],
    "resnet50": [2048, 2048, 1024, 512, 256],
}


def get_encoder(config):
    if config['encoder_type'] == "vgg19":
        encoder = VGG19()
    elif config['encoder_type'] == "vgg19_bn":
        encoder = VGG19_bn()
    elif config['encoder_type'] == "resnet152":
        encoder = ResNet152()
    elif config['encoder_type'] == "resnet34":
        encoder = ResNet34()
    elif config['encoder_type'] == "resnet50":
        encoder = ResNet50()
    else:
        raise RuntimeError("invalid encoder type")

    return encoder

def get_fpn_skip(config):
    return layer_channels[config['encoder_type']][::-1]
