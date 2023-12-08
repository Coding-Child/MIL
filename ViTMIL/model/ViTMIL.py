import torch
import torch.nn as nn
from model.ViTAdapter import ViT


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


class ViTMIL(nn.Module):
    def __init__(self, pretrained: bool = True,
                 progress: bool = False,
                 key: str = 'DINO_p16',
                 patch_size: int = 16,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super(ViTMIL, self).__init__()

        self.vit = ViT(pretrained=pretrained, progress=progress, key=key, patch_size=patch_size)
        self.mha = nn.MultiheadAttention(embed_dim=384, dropout=dropout, num_heads=num_heads)
        self.pool = GeM()
        self.norm = nn.LayerNorm(384)
        self.ffn = nn.Sequential(nn.Linear(384, 1536),
                                 nn.GELU(),
                                 nn.Linear(1536, 384),
                                 )
        self.classifier = nn.Linear(384, 1)

    def forward(self, x):
        src, out_instance = self.vit(x)
        attn_mask = create_padding_mask(src)
        src, _ = self.mha(src, src, src, key_padding_mask=attn_mask)

        out = self.norm(src)
        out = self.ffn(out)
        out = self.pool(out)
        out = self.classifier(out)

        out_instance = out_instance.squeeze()
        out = out.squeeze()

        return out_instance, out


if __name__ == '__main__':
    from torchsummary import summary

    model = ViTMIL().cuda()
    summary(model, (32, 3, 224, 224), batch_size=16)
