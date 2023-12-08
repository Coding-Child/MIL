import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet

from einops import repeat


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_pretrained_url(key):
    url_prefix = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{url_prefix}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )

        print(verbose)
    return model


def always_evaluate(model_method):
    def wrapper(*args, **kwargs):
        self = args[0]
        self.eval()
        with torch.no_grad():
            return model_method(*args, **kwargs)
    return wrapper


def remove_prefix(state_dict, prefix):
    return {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}

# 불러온 state_dict를 수정합니다.


class ResNet50(nn.Module):
    def __init__(self, path):
        super(ResNet50, self).__init__()

        self.model = resnet50(pretrained=False, progress=False, key=None)
        self.model.load_state_dict(remove_prefix(torch.load(path)['state_dict'], 'model.'))

        for param in self.model.parameters():
            param.requires_grad = False

        self.extractor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Flatten()
                                       )

    @always_evaluate
    def forward(self, x):
        out = self.model(x)
        out = self.extractor(out)

        return out


def modified_prefix(state_dict, prefix):
    return {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}


class ResNetMIL(nn.Module):
    def __init__(self, d_model, path=None, pretrained=False):
        super(ResNetMIL, self).__init__()

        self.d_model = d_model
        self.model = resnet50(pretrained=pretrained, progress=False, key='MoCoV2')

        if path is not None:
            self.model.load_state_dict(modified_prefix(torch.load(path)['state_dict'], 'model.'))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.init_weights()
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten())

        self.query_layer = nn.Linear(3840, d_model)
        self.encoding_layer = nn.Linear(2048, d_model)
        self.instance_classifier = nn.Linear(2048, 1)

        self._register_hooks()

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)

        instance_feature = self.pooling(self.model(x))
        instance_logits = self.instance_classifier(instance_feature).view(b, n)

        instance_feature = self.encoding_layer(instance_feature)
        features = instance_feature.view(b, n, self.d_model)

        feat1 = self.pooling(self.feature_maps['layer2'].to(x.device)).view(b, n, -1)
        feat2 = self.pooling(self.feature_maps['layer3'].to(x.device)).view(b, n, -1)
        feat3 = self.pooling(self.feature_maps['layer4'].to(x.device)).view(b, n, -1)
        feat4 = self.pooling(self.feature_maps['layer5'].to(x.device)).view(b, n, -1)

        query = torch.cat((feat1, feat2, feat3, feat4), dim=-1)
        query = self.query_layer(query)
        q = torch.cat((cls_tokens, query), dim=1)

        features = torch.cat((cls_tokens, features), dim=1)

        return q, features, instance_logits

    def _register_hooks(self):
        self.feature_maps = {}

        def hook_fn(name):
            def hook(model, input, output):
                self.feature_maps[name] = output

            return hook

        # 각 레이어에 훅을 등록합니다.
        self.model.layer1.register_forward_hook(hook_fn('layer2'))
        self.model.layer2.register_forward_hook(hook_fn('layer3'))
        self.model.layer3.register_forward_hook(hook_fn('layer4'))
        self.model.layer4.register_forward_hook(hook_fn('layer5'))

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


if __name__ == '__main__':
    from torchsummary import summary
    # import torch.nn.functional as F

    model = ResNet50(path='../simclr_pretrained_model_ckpt/checkpoint_0200_Adam.pt').cuda()
    summary(model, (3, 224, 224))
    # input = torch.rand((1, 56, 3, 224, 224)).cuda()
    #
    # print(model(input).shape)
    # kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    # kl_loss = kl_div_loss(F.log_softmax(torch.rand(1, 56, 2), dim=-1), F.softmax(torch.rand(1, 56, 2), dim=-1))
    # print(kl_loss)
