import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer


# Adapter 모듈 정의
class AdapterModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdapterModule, self).__init__()

        self.up_project = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.down_project = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Skip connection을 위해 입력값 저장
        original_x = x
        # Adapter의 Feedforward up-project
        x = self.up_project(x)
        # Nonlinearity 적용
        x = self.activation(x)
        # Adapter의 Feedforward down-project
        x = self.down_project(x)
        # Adapter의 출력과 원래 입력값을 더함
        return original_x + x


class AdapterBlock(nn.Module):
    def __init__(self, block, adapter):
        super(AdapterBlock, self).__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x):
        x = self.block(x)
        x = self.adapter(x)
        return x


# ViT 모델에 Adapter 추가하는 함수
def add_adapters_to_vit(model, hidden_dim=64):
    for i, block in enumerate(model.blocks):
        ffn_dim = block.mlp.fc2.out_features  # FFN의 출력 차원을 가정함
        adapter = AdapterModule(ffn_dim, hidden_dim)
        # block.mlp에 adapter를 직접 추가하는 대신, TransformerBlockWithAdapter를 사용하여 감쌈
        adapted_block = AdapterBlock(block, adapter)
        # blocks의 해당 위치에 새로운 adapted_block을 설정
        model.blocks[i] = adapted_block


# 모델을 로드하고 Adapter를 추가하는 함수
def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=0
    )
    if pretrained:
        URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
        model_zoo_registry = {
            "DINO_p16": "dino_vit_small_patch16_ep200.torch",
            "DINO_p8": "dino_vit_small_patch8_ep200.torch",
        }
        pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )

    # Adapter 추가
    add_adapters_to_vit(model, hidden_dim=64)

    for name, param in model.named_parameters():
        if 'adapter' in name:  # Adapter 모듈을 제외한 모든 파라미터를 고정
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


class ViT(nn.Module):
    def __init__(self, pretrained, progress, key, patch_size=16):
        super(ViT, self).__init__()

        self.vit = vit_small(pretrained, progress=progress, key=key, patch_size=patch_size)
        self.classifier = nn.Linear(in_features=384, out_features=1)

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        x_features = self.vit(x)
        x = self.classifier(x_features)

        x_features = x_features.view(b, n, -1)
        x = x.view(b, n, -1)

        return x_features, x


if __name__ == "__main__":
    model = ViT(pretrained=True, progress=False, key="DINO_p16", patch_size=16).cuda()
    data = torch.rand((1, 16, 3, 224, 224)).cuda()

    features, out = model(data)
    print(features.shape)
    print(out.shape)
