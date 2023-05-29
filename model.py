import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, config, out):
        super(VGG16, self).__init__()
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = models.vgg16(weights=weights)
        print(self.net)
        # remove the last FC layer
        num_output_feats = self.net.classifier[-1].in_features  # dim  of the features
        # Initialize a new fully connected layer
        self.net.classifier[-1] = torch.nn.Linear(num_output_feats, out)

        for i, param in enumerate(self.net.features.parameters()):
            if i < 10:
                param.requires_grad = False
        self.base = self.net
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        output = self.logSoftmax(x)
        return output

    def get_feature_layer(self, x):
        x = self.base(x)
        # x = self.fc1(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, config, out):
        super(ResNet50, self).__init__()
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = models.resnet50(weights=weights)
        print(self.net)
        # remove the last FC layer
        num_output_feats = self.net.fc.in_features   # dim  of the features
        # Initialize a new fully connected layer
        self.net.fc = torch.nn.Linear(num_output_feats, out)

        for i, param in enumerate(self.net.parameters()):
            if i < 10:
                param.requires_grad = False
        self.base = self.net
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        output = self.logSoftmax(x)
        return output

    def get_feature_layer(self, x):
        x = self.base(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, config, out):
        super(EfficientNet, self).__init__()
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = models.efficientnet_b5(weights=weights)
        print(self.net)
        # remove the last FC layer
        num_output_feats = self.net.classifier[-1].in_features  # dim  of the features
        # Initialize a new fully connected layer
        self.net.classifier[-1] = torch.nn.Linear(num_output_feats, out)

        for i, param in enumerate(self.net.parameters()):
            if i < 10:
                param.requires_grad = False
        self.base = self.net
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        output = self.logSoftmax(x)
        return output

    def get_feature_layer(self, x):
        x = self.base(x)
        return x
