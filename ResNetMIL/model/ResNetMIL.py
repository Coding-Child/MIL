import torch
import torch.nn as nn
from model.ResNetAdapter import ResNet50


def create_padding_mask(features):
    mask = torch.all(features == 0, dim=-1)
    mask = mask.transpose(1, 0)

    return mask


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=1).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class ResNetMIL(nn.Module):
    def __init__(self,
                 pretrained: bool = True,
                 progress: bool = False,
                 key: str = 'MoCoV2',
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super(ResNetMIL, self).__init__()

        self.resnet = ResNet50(pretrained=pretrained, progress=progress, key=key)
        self.mha = nn.MultiheadAttention(embed_dim=2048, dropout=dropout, num_heads=num_heads)
        self.pool = GeM()
        self.norm = nn.LayerNorm(2048)
        self.ffn = nn.Sequential(nn.Linear(2048, 4096),
                                 nn.GELU(),
                                 nn.Linear(4096, 2048)
                                 )
        self.reducer = nn.Sequential(nn.Linear(3840, 2048),
                                     nn.ReLU(inplace=True)
                                     )
        self.classifier = nn.Sequential(nn.Linear(2048, 4096),
                                        nn.Linear(4096, 4096),
                                        nn.Linear(4096, 1)
                                        )

    def forward(self, x):
        b, n, _, _, _ = x.size()
        src, out_instance, feature_map = self.resnet(x)
        attn_mask = create_padding_mask(src)

        pooled_outputs = []
        for key in ["conv_2x", "conv_3x", "conv_4x", "conv_5x"]:
            pooled_output = nn.functional.adaptive_avg_pool2d(feature_map[key], (1, 1))
            pooled_output = pooled_output.view(pooled_output.size(0), -1)
            pooled_outputs.append(pooled_output)
        pooled_outputs = torch.cat(pooled_outputs, dim=1)
        query = self.reducer(pooled_outputs).view(b, n, -1)

        src, _ = self.mha(query, src, src, key_padding_mask=attn_mask)

        out = self.norm(src)
        out = self.ffn(out)
        out = self.pool(out)
        out = self.classifier(out)
        out = out.squeeze()

        return out_instance, out


if __name__ == '__main__':
    model = ResNetMIL(pretrained=True, progress=False, key="MoCoV2").cuda()
    data = torch.rand((8, 32, 3, 224, 224)).cuda()
    out_1, out_2 = model(data)

    # print(out_1.shape)
    # print(out_2.shape)
    # print(model)
