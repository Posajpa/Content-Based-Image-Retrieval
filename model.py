import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, config, out):
        super(VGG, self).__init__()
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = models.vgg16(weights=weights)
        # remove the last FC layer
        num_output_feats = self.net.classifier[-1].in_features  # dim  of the features
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