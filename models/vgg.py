import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg19_adding_conv1x1_f', 'vgg19_adding_conv1x1_g', 'vgg19_adding_conv1x1_h', 'vgg19_adding_conv1x1_h_removing_1conv3x3', 'vgg19_removing_1conv3x3',
    'vgg19_started_resnet0', 'vgg19_started_resnet', 'vgg19_started_resnet2', 'vgg19_started_resnet3', 'vgg19_started_resnet4', 'vgg19_started_resnet5', 'vgg19_started_resnet6', 'vgg19_started_resnet7',
    'vgg19_started_resnet8', 'vgg19_started_resnet9', 'vgg19_started_resnet10', 'vgg19_started_resnet11', 'vgg19_started_resnet12', 'vgg19_started_resnet13', 'vgg19_started_resnet14',
    'vgg19_started_resnet15', 'vgg19_started_resnet16', 'vgg19_started_resnet17', 'vgg19_started_resnet18', 'vgg19_started_resnet19', 'vgg19_started_resnet20', 'vgg19_started_resnet21',
    'vgg19_started_resnet22'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG_CONV1X1(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_CONV1X1, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_adding_conv1x1(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'conv1x1':
            conv2d_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            layers += [conv2d_1x1, nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG_STARTED_RESNET(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_STARTED_RESNET, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers_started_resnet_wo_bn(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'conv7x7':
            conv2d_7x7 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            layers += [conv2d_7x7, nn.ReLU(inplace=True)]
            in_channels = 64
        elif v == 'conv1x1':
            conv2d_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            layers += [conv2d_1x1, nn.ReLU(inplace=True)]
        elif v == 'conv1x1_64to128':
            conv2d_1x1 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
            layers += [conv2d_1x1]
            in_channels = 128
        elif v == 'conv1x1_128to256':
            conv2d_1x1 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
            layers += [conv2d_1x1]
            in_channels = 256
        elif v == 'conv1x1_256to512':
            conv2d_1x1 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
            layers += [conv2d_1x1]
            in_channels = 512
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_started_resnet(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'conv7x7':
            conv2d_7x7 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            layers += [conv2d_7x7, nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            in_channels = 64
        elif v == 'conv1x1':
            conv2d_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            layers += [conv2d_1x1, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
        elif v == 'conv1x1_64to128':
            conv2d_1x1 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
            layers += [conv2d_1x1, nn.BatchNorm2d(128)]
            in_channels = 128
        elif v == 'conv1x1_128to256':
            conv2d_1x1 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
            layers += [conv2d_1x1, nn.BatchNorm2d(256)]
            in_channels = 256
        elif v == 'conv1x1_256to512':
            conv2d_1x1 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
            layers += [conv2d_1x1, nn.BatchNorm2d(512)]
            in_channels = 512
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 64, 'conv1x1', 'M', 128, 128, 'conv1x1', 'M', 256, 256, 256, 256, 'conv1x1', 'M', 512, 512, 512, 512, 'conv1x1', 'M', 512, 512, 512, 512, 'conv1x1', 'M'],
    'G': [64, 64, 'M', 128, 'conv1x1', 128, 'conv1x1', 'M', 256, 'conv1x1', 256, 'conv1x1', 256, 'conv1x1', 256, 'conv1x1', 'M', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M'],
    'H': [64, 64, 'M', 'conv1x1', 128, 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 256, 'conv1x1', 256, 'conv1x1', 256,
          'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 512,
          'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M'],
    'I': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'H2': [64, 'M', 'conv1x1', 128, 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 256, 'conv1x1', 256,
          'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M', 'conv1x1',
          512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M'],
    'J0': ['conv7x7', 'M2', 'conv1x1_64to128', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1_128to256', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1_256to512', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J': ['conv7x7', 'conv1x1_64to128', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1_128to256', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1_256to512', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J2': [64, 64, 'M', 'conv1x1_64to128', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1_128to256', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1_256to512', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J3': [64, 64, 'M', 'conv1x1', 64, 'conv1x1', 'conv1x1', 64, 'conv1x1', 'M', 'conv1x1_64to128', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128,
           'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1_128to256', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1',
           'M', 'conv1x1_256to512', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J4': ['conv7x7', 'M2', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J5': [64, 64, 'M', 'conv1x1', 64, 'conv1x1', 'conv1x1', 64, 'conv1x1', 'M', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128,
           'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J6': [64, 64, 'M', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J7': [64, 64, 'M', 'conv1x1', 128, 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 256, 'conv1x1', 256,
          'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M', 'conv1x1',
          512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 512, 'conv1x1', 'M'],
    'J8': [64, 64, 'M', 'conv1x1_64to128', 128, 128, 'M', 'conv1x1_128to256', 256, 256, 256, 256, 'M', 'conv1x1_256to512', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J9': [64, 64, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'J10': ['conv7x7', 'M2', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J11': ['conv7x7', 'M2', 'conv1x1_64to128', 128, 128, 'M', 'conv1x1_128to256', 256, 256, 256, 256, 'M', 'conv1x1_256to512', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J12': [64, 64, 'M', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J13': [64, 64, 'M', 'conv1x1', 128, 128, 'conv1x1', 'M', 'conv1x1', 256, 256, 256, 256, 'conv1x1', 'M', 'conv1x1', 512, 512, 512, 512, 'conv1x1', 'M', 'conv1x1', 512, 512, 512, 512, 'conv1x1', 'M'],
    'J14': [64, 64, 'M', 'conv1x1', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J15': ['conv7x7', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J16': ['conv7x7', 'conv1x1_64to128', 128, 128, 'M', 'conv1x1_128to256', 256, 256, 256, 256, 'M', 'conv1x1_256to512', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J17': ['conv7x7', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J18': [64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'J19': [64, 64, 'M', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J20': [64, 64, 'M', 'conv1x1_64to128', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1_128to256', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1_256to512', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
    'J21': [64, 64, 'M', 'conv1x1', 128, 'conv1x1', 'conv1x1', 128, 'conv1x1', 'M', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256, 'conv1x1', 'conv1x1', 256,
           'conv1x1', 'conv1x1', 256, 'conv1x1', 'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1',
           'M', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'conv1x1', 512, 'conv1x1', 'M'],
}


# J performs bad, J's architecture is almost equal to resnet50_wo_skip2, they all perform bad
# J2 performs better, but not good
# J3 performs better, but not good  by comparing J and J2, J3, we conclude that 64, 64, M better than conv7x7 used in resnet
# J4 performs worse than J3, better than J
# J5 performs like J4
# J6 performs better than J5, we conclude that enhancing channels number is helpful
# J7 performs like J6, we conclude that inserting multiple conv1x1 without channels change make little difference


# 0) by comparing resnet50_plain3 and resnet50_plain5, resnet50_plain4 and resnet50_plain6, we conclude residual connection can always largely degrade performance
# 1) by comparing J7 and E, we conclude that solely inserting conv1x1 without channel number change can degrade performance, but not too much
# 2) by comparing J8 and E, we conclude that solely inserting conv1x1 with channel number change can degrade performance, but not too much
# 3) by comparing J9 and E with bn, we conclude that solely enhance the channel number can make performance better, but a little bit
# 4) by comparing J6 and E with bn, we conclude that although solely inserting conv1x1 without channel number change doen't
#    make performance worse a lot, but can make the model unrobust, saying, further adding conv1x1 with channel number (J3 vs J6) or conv7x7 (J4 vs J6) or both (J vs J6) can degrade performance a lot.



def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model



def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model



def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model



def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model



def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model



def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model



def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


def vgg19_adding_conv1x1_f(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_CONV1X1(make_layers_adding_conv1x1(cfg['F']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def vgg19_adding_conv1x1_g(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_CONV1X1(make_layers_adding_conv1x1(cfg['G']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def vgg19_adding_conv1x1_h(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_CONV1X1(make_layers_adding_conv1x1(cfg['H']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def vgg19_removing_1conv3x3(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['I']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def vgg19_adding_conv1x1_h_removing_1conv3x3(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_CONV1X1(make_layers_adding_conv1x1(cfg['H2']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_started_resnet0(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J0'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


def vgg19_started_resnet(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet2(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J2'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet3(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J3'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet4(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J4'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet5(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J5'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet6(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J6'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet7(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J7'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet8(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J8'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet9(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J9'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet10(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J10'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet11(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J11'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet12(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J12'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet13(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J13'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet14(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J14'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet15(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J15'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet16(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J16'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


def vgg19_started_resnet17(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J17'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet18(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J18'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J19'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


def vgg19_started_resnet20(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet(cfg['J20'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def vgg19_started_resnet21(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_STARTED_RESNET(make_layers_started_resnet_wo_bn(cfg['J21'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def vgg19_started_resnet22(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model