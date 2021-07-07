---
layout: post
title: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
date: 2021-01-25
category: review
thumbnail: /style/image/VGG.png
use_math: true
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Karen Simonyan, Andrew Zisserman</span>

# Abstact

- **본 논문에서는 large-scale image recognition setting에서 CNN의 깊이가 accuarcy에 미치는 영향에 관한 연구 진행**
- **주요 기여**: 매주 작은 (3x3) convolution filter를 사용해서 깊이를 증가시키면서 network를 평가 (16-19 weight layer를 쌓아서 이전 보다 상당히 개선함)
- 다른 데이터 셋에서도 일반화되어 가장 좋은 성능을 얻을 수 있었고, 두 가지 최고 성능의 ConvNet 모델을 공개

# 1. Introduction

- Convolutional Networks (ConvNets)은 최근 large-scale 이미지 및 비디오 인식에서 아주 좋은 성능을 보임
    - deep ConvNets(2012)을 개선하기 위한 여러 시도를 진행
        1. 첫 번째 convolutional layer의 strid와 window size를 더 작게 사용하는 것(2013)
        2. 전체 이미지와 여러  크기에 걸쳐 조밀하게 network를 훈련하고 테스트하는 것(2014)
- 본 논문에서는 ConvNet architecture 설계의 또 다른 중요한 측면인 "**깊이, depth**"에 대해 다룸
    - 이를 위해 **모든 layer에서 (3x3)의 매주 작은 convolution filter를 사용**하여 network의 깊이를 계속 증가시킴
    — parameter 수를 줄임으로써 일반화가 더 용이, overfitting을 막고, 연산량을 줄임
    - 결과적으로 classification과 **localisation tasks**에 대한 가장 좋은 정확도 뿐만 아니라 다른 이미지 인식 데이터 셋에도 적용할 수 있는 훨씬 더 정확한 ConvNet architecture를 개발

**localisation tasks**
object가 있는 위치를 찾아 그 주위에 bounding box를 그리는 것

❓❓❓

**large public image repositories**
기존에는 이미지 많지 않아서 훈련이 크게, 많이 할 수 없었는데 ImageNet(image database)과 같은 저장소가 생겨서 이러한 문제점을 해결할 수 있었음
**high-dimensional shallow feature**
높은 차원의 데이터를 학습 계층을 적게 사용해서도 학습을 가능하게 했다? — 딥러닝이 유행하기 전에는 자동으로 네트워크를 훈련하는게 아니라, 필터를 수동으로 손으로 만들어 사용했는데 상대적으로 dimension이 크고 복잡한데 예전 모델을 일컫는 것 같음

**testbed**
image를 활용해서 내가 가지고 있는 문제나 모델을 테스트하는 장소

**used as a part of a relatively simple pipelines**

다른 모델과 합쳐서 사용할 때 앞단에 많이 사용됨

# 2. ConvNet Configurations

### 2.1 Architecture

> **Conv layer**

- **feature extractor**
- input: fixed-size 224x224 RGB (preprocessing: 각 channel에 대해 mean빼는 것 — data centering)
    - 음수~양수로 값의 범위를 맞춤
- (3x3) filter를 사용하는 convolutional layer를 쌓은 구조
    - 3x3 filter: 위/아래, 왼쪽/오른쪽, 중앙의 정보를 수집할 수 있는 가장 작은 크기
- 1x1 convolution filter도 사용
    - input channels의 linear transformation을 위해
- stride: 1, padding 적용 O

> **Spatial Pooling layer**

- conv layer 이후 적용 (모든 conv 이후에 사용되지는 않음)
- 총 5개의 max poolinhg layer 사용
    - 2x2 size, stride: 2

❓❓❓

**Spatial Pooling layer = Spatial Pyramid Pooling? No!**

~~만약 두개가 같다면 이미지 인식에서 일정한 크기로 자르거나 축소해서 모델에 넣은게 아니라 통채로 넣고 pooling을 이용해서 일정한 크기로 맞추서 FC 입력으로 넣는 것 같은데 뒤에서 훈련할 때 특정 차원으로 맞추는 것 같은데 왜 이 방법은 사용하는 것인지?~~

→ 10x10이 있으면 이걸 줄여서 stride에 맞춰 5x5로 줄이는 것 (일반적인 pooling이랑 같음)

> **Fully-connected layer (FC)**

- **conv에서 나온 feature로 확률값을 이용해 classification**
- 3개의 FC 사용 + softmax layer
    - 1-2 layer: 4096개 node, 3 layer: 1000 (classification을 위해)
- activation function: ReLU
- Local Response Normalization (LRN) 정규화 포함X (하나 제외하고)
    - ReLU를 사용하면 양수값은 자기 자신이 나오게 되어, 매우 큰 값을 갖는 경우(outlier) 다른 값들이 기능을 못할 수 있음
    - ReLU 이후에 나오는 값을 주변 값을 이용해 normalize해줌으로써 이러한 것을 완화

### 2.2 Confiurations

- 깊이가 더 깊어짐에도 불구하고 본 논문에서 제안하는 network에 있는 가중치의 수는 더 얕고 큰 conv를 갖는 모델의 가중치 수보다 크지 않음
    - Sermanet et al., 2014: 144M weights

<center><img src="https://user-images.githubusercontent.com/46676700/124697233-67018600-df21-11eb-86c6-962f45358b86.png" alt="img" style="zoom: 80%;" /></center>

### 2.3 Discussion

> **차별점**

- stride=4 11x11 filter, stride=2 7x7 filter와 같이 큰 filter를 사용하는 이전 모델들과 달리 **3x3의 매우 작은 size의 filter를 사용**

> **3x3 filter를 사용하는 이유**

Table1에서 중간에 spatial pooling이 없는 경우 여러 개의 conv가 stack되어 있는 것을 알 수 있음

- 2개를 사용하게되면 5x5 filter를 사용하는 것과 같은 기능을 수행할 수 있음 (= effective receptive field가 같음)
    - 3개 사용: 7x7

**→ 얻는 이점은?**

1. activation function을 더 많이 거치면서 non-linear한 문제를 더 잘 풀 수 있게 됨
2. parameter의 수를 줄임
    - C채널의 3x3 convolution이 3 layer인 경우: $3(3^2C^2) = 27C^2$
    - C채널 7x7 convolution이 1 layer인 경우: $7^2(C^2)=49C^2$

+ decision function의 비선형성을 증가시키기 위해 1x1 conv를 사용
— 비선형성을 증가시키면 좀 더 복잡한 문제를 풀 수 있게 됨

> **유사한 task**

- Lin et al.(2014)
  — "Network in Network"에서 1x1 conv가 활용됨, 그러나 본 논문의 구조보다 깊지 않으며 ILSVRC 데이터 셋에서 평가하지 않음  

- Goodfellow et al.(2014)
  — 거리 번호 인식에서 깊은 ConvNets을 적용했고 깊이가 증가함에 따라 성능이 향상됨을 보여줌  

- Szegedy et al.(2014)
  —"GoogLeNet" 매우 깊은 ConvNet(22 layer)와 작은 convolution을 기반한다는 점에서 유사함 (1x1, 5x5 사용), 본 논문보다 network topology가 복잡하고 단일 네트워크 분류에서 본 논문 성능이 더 우수

# 3. Classification Framework

### 3.1 Training

> **hyperparameter**

- cost function: Cross Entropy
- mini-batch size: 256
- optimizer: Momentum=0.9
- regularization: L2 regularization($5 · 10^{−4}$), Dropout(0.5)
- learning rate: $10^{-2}$ (validation accuarcy의 증가가 멈추면 0.1씩 감소 — 3배 감소)
- 370L iterations (74 epochs)
- pre-initialization: A model의 일부(처음 4개 conv+마지막 3개 FC)를 훈련한 뒤 가져와서 초기값으로 사용

> **Training image size**

**isotropically-rescaled**

- image를 VGG model input size(224x224)에 맞도록 변경해줘야 함
- **S를 이용**해서 **비율은 그대로** 두고 size를 바꾼 뒤 **crop하여 사용**

**training scale S**

1. S를 고정시키는 것
    - S=256으로 두어 먼저 network를 훈련하고, S=384로 훈련할 때는 256으로 훈련한 파라미터로 가중치를 초기화하여 사용하고, 더 작은 learning rate 사용 ($10^{-3}$)

2. 256~512 중 random하게 S값 사용 (multi-scale)
    - object가 모두 다른 size를 갖으면서 학습효과가 더 좋아질 수 있음
    - data augmentation 효과(= scale jittering)

### 3.2 **Testing**

train에 사용된 S와 같은 역할을 하는 **Q** 를 사용하여 **image rescaling** 적용

- $Q \ne S$

**구조 변경** (crop하지 않은 전체 이미지에 적용할 수 있음)

- FC layer → conv
    - first: **7x7 conv**
    - last two: **1x1 conv**

→ class 수와 동일한 channel 수와 input image size에 따라 가변 공간 해상도를 갖는 class score map

**고정 크기의 벡터**

- class score를 얻기 위해 pooling 진행(spatially averaged)
- image를 수평으로 뒤집어서, 원본 이미지와 뒤집힌 이미지의 softmax 결과를 평균내 최종 score로 사용

---

**Reference**

- paper [[📑](https://arxiv.org/pdf/1409.1556.pdf)]
- CNN의 parameter 개수와 tensor 사이즈 계산하기 [[👆](https://seongkyun.github.io/study/2019/01/25/num_of_parameters/)]
