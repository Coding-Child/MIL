# MIL

Multiple Instance Learning을 위한 코드

- ViTMIL
  * Instance Encoding을 ViT를 이용 &rightarrow; Pre-trained 된 ViT를 가져와 Parameter Freezing을 한 후 Feed Forward Network 뒤에 학습 가능한 Adapter 결합
  * ViT를 거친 후 Transformer의 Multi-Head Self-Attention을 이용하여 각 Instance 간의 Attention 진행
  * GeM pooling 이용하여 Sequence의 각 feature를 통합 Shape: (b, seq_len, embed_dim) &rightarrow; (b, embed_dim)
  * classifier를 통해 classification 진행
 
- ResNetMIL
  - ResNet50을 Instance Encoder로 사용
  - ResNet50을 거친 후 Transformer Encoder Layer에 들어가게 됨
  - Transformer Encoder Layer를 거쳐 CLS 토큰의 feature 추출
  - CLS 토큰의 feature를 classifier를 이용하여 예측 생성
