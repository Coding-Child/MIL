import torch
import torch.nn as nn

from model.Transformer import TransformerEncoder
from model.ResNet import ResNetMIL


def create_padding_mask(features):
    mask = torch.all(features == 0, dim=-1)
    mask = mask.to(torch.bool)
    mask = mask.transpose(0, 1)

    return mask


class RTMIL(nn.Module):
    def __init__(self, d_model, d_ff, dropout, num_heads, num_layers, num_classes, path, pretrained):
        super(RTMIL, self).__init__()

        self.num_classes = num_classes

        self.resnet = ResNetMIL(d_model=d_model, path=path, pretrained=pretrained)
        self.transformer = TransformerEncoder(d_model, d_ff, dropout, num_heads, num_layers)

        if self.num_classes == 2:
            self.bag_classifier = nn.Linear(d_model, 1)
        else:
            self.bag_classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        query, features, instance_logits = self.resnet(x)

        padding_mask = create_padding_mask(features)

        out = self.transformer(query, features, padding_mask)
        out = out[:, 0]

        if self.num_classes == 2:
            bag_logits = self.bag_classifier(out).squeeze()
        else:
            bag_logits = self.bag_classifier(out)

        return bag_logits, instance_logits
