import torch
from torchvision.models import resnet152, resnet34, resnet50, resnet101

# __all__ = ['ResNet50', 'ResNet101','ResNet152']
#
# def Conv1(in_planes, places, stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=3,stride=stride,padding=3, bias=False),
#         nn.BatchNorm2d(places),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )
#
# class Bottleneck(nn.Module):
#     def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
#         super(Bottleneck,self).__init__()
#         self.expansion = expansion
#         self.downsampling = downsampling
#
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
#             nn.BatchNorm2d(places),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(places),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(places*self.expansion),
#         )
#
#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(places*self.expansion)
#             )
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self, x):
#         residual = x
#         out = self.bottleneck(x)
#
#         if self.downsampling:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#         return out
#
# class ResNet(nn.Module):
#     def __init__(self,blocks, num_classes=1000, expansion = 4):
#         super(ResNet,self).__init__()
#         self.expansion = expansion
#
#         self.conv1 = Conv1(in_planes = 3, places= 64)
#
#         self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
#         self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
#         self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
#         self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def make_layer(self, in_places, places, block, stride):
#         layers = []
#         layers.append(Bottleneck(in_places, places,stride, downsampling =True))
#         for i in range(1, block):
#             layers.append(Bottleneck(places*self.expansion, places))
#
#         return nn.Sequential(*layers)
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#         skip_1 = x
#         x = self.layer1(x)
#         skip_2 = x
#         x = self.layer2(x)
#         skip_3 = x
#         x = self.layer3(x)
#         skip_4 = x
#         x = self.layer4(x)
#         return skip_1, skip_2, skip_3, skip_4, x
#
#
# def ResNet50():
#     return ResNet([3, 4, 6, 3])
#
#
# def ResNet101():
#     return ResNet([3, 4, 23, 3])
#
#
# def ResNet152():
#     return ResNet([3, 8, 36, 3])


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet152(torch.nn.Module):
    def __init__(self):
        super(ResNet152,self).__init__()
        resnet = resnet152(pretrained=True)
        self.inconv = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.inbn = torch.nn.BatchNorm2d(64)
        self.inrelu = torch.nn.ReLU(inplace=True)
        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(2048, 2048)
        self.resb5_2 = BasicBlock(2048, 2048)
        self.resb5_3 = BasicBlock(2048, 2048)  # 14

    def forward(self, x):
        x = self.inconv(x)
        x = self.inbn(x)
        x = self.inrelu(x)

        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        x = self.pool4(skip4)
        x = self.resb5_1(x)
        x = self.resb5_2(x)
        x = self.resb5_3(x)
        return skip1, skip2, skip3, skip4, x

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50,self).__init__()
        resnet = resnet50(pretrained=True)
        self.inconv = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.inbn = torch.nn.BatchNorm2d(64)
        self.inrelu = torch.nn.ReLU(inplace=True)
        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(2048, 2048)
        self.resb5_2 = BasicBlock(2048, 2048)
        self.resb5_3 = BasicBlock(2048, 2048)  # 14

    def forward(self, x):
        x = self.inconv(x)
        x = self.inbn(x)
        x = self.inrelu(x)

        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        x = self.pool4(skip4)
        x = self.resb5_1(x)
        x = self.resb5_2(x)
        x = self.resb5_3(x)
        return skip1, skip2, skip3, skip4, x

class ResNet34(torch.nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        resnet = resnet34(pretrained=True)
        self.inconv = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.inbn = torch.nn.BatchNorm2d(64)
        self.inrelu = torch.nn.ReLU(inplace=True)
        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14

    def forward(self, x):
        x = self.inconv(x)
        x = self.inbn(x)
        x = self.inrelu(x)

        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        x = self.pool4(skip4)
        x = self.resb5_1(x)
        x = self.resb5_2(x)
        x = self.resb5_3(x)
        return skip1, skip2, skip3, skip4, x

