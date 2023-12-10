import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet


class ConvAdapter(nn.Module):
    def __init__(self, input_channels, output_channels, alpha=1.0, beta=0.1):
        super(ConvAdapter, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.depthwise_conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)
        self.pointwise_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = F.relu(x)
        x = self.pointwise_conv(x)
        x = self.alpha * x + self.beta * residual
        return x


class ResNetTrunk(ResNet):
    def __init__(self, block, layers, *args, **kwargs):
        super().__init__(block, layers, *args, **kwargs)
        del self.fc  # FC layer 제거
        # 각 ResNet layer 후에 adapter 적용
        self.adapter1 = ConvAdapter(256, 256)  # 예시 채널 크기, 실제 값으로 바꿔야 함
        self.adapter2 = ConvAdapter(512, 512)
        self.adapter3 = ConvAdapter(1024, 1024)
        self.adapter4 = ConvAdapter(2048, 2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        output = dict()
        x = self.layer1(x)
        output["conv_2x"] = x
        x = self.adapter1(x)  # 첫 번째 adapter 적용
        x = self.layer2(x)
        output["conv_3x"] = x
        x = self.adapter2(x)  # 두 번째 adapter 적용
        x = self.layer3(x)
        output["conv_4x"] = x
        x = self.adapter3(x)  # 세 번째 adapter 적용
        x = self.layer4(x)
        output["conv_5x"] = x
        x = self.adapter4(x)  # 네 번째 adapter 적용
        return x, output


def freeze_parameters(model, adapters):
    # 모델의 모든 파라미터를 freeze
    for param in model.parameters():
        param.requires_grad = False
    # Adapter 파라미터만 unfreeze
    for adapter in adapters:
        for param in adapter.parameters():
            param.requires_grad = True


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress),
                              strict=False
                              )

    freeze_parameters(model, [model.adapter1, model.adapter2, model.adapter3, model.adapter4])
    return model


class ResNet50(nn.Module):
    def __init__(self, pretrained, progress, key, **kwargs):
        super().__init__()
        self.trunk = resnet50(pretrained, progress, key, **kwargs)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten()
                                  )
        self.classfier = nn.Sequential(nn.Linear(2048, 4096),
                                       nn.Linear(4096, 4096),
                                       nn.Linear(4096, 1)
                                       )

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(b * n, c, h, w)

        feautures, feature_map = self.trunk(x)
        feautures = self.pool(feautures)
        out = self.classfier(feautures)

        feautures = feautures.view(b, n, -1)
        out = out.view(b, n)

        return feautures, out, feature_map


if __name__ == '__main__':
    from torchsummary import summary

    model = ResNet50(pretrained=True, progress=True, key="MoCoV2").cuda()
    summary(model, (16, 3, 224, 224))
