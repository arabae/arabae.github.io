---
layout: post
title: "Cross attentive pooling for speaker verification"
date: 2020-10-13
category: review
thumbnail: /style/image/CAP.png
use_math: true
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Seong Min Kye, Yoohwan Kwon, Joon Son Chung</span>

# 📌 **Abstract**

- **목표 : 'in the wild' video와 관련없는 signal을 포함하는 utterance를 사용하는 TI-SV(Text-Independent Speaker Verification)**  
- SV는 pair-wise 문제(등록과 테스트 쌍을 비교), 기존의 embedding 추출은 instance-wise 문제(각 utterance에 대한 embedding을 추출하여 서로 비교)
- <span style="background-color:#ffed54">본 논문에서는 reference-query pair 전체의 context 정보를 활용하여 **pair-wise 문제에 가장 discriminative한 utterance-level의 embedding 추출을 생성**하는 **CAP(Cross Attention Pooling)**을 제안</span>
- VoxCeleb dataset을 사용하고, 다른 pooling 방법과 비교하여 우수한 성능을 보였음

<br/>

---

<br/>


# **Ⅰ. Introduction**

- Automatic Speaker Recognition; 음성은 가장 쉽게 접근할 수 있는 생체 정보 중 하나이기 때문에 누군가의 신원을 확인하는데 매력적인 방법
- speaker recognition은 identification과 verification을 모두 포함하지만, 후자의 경우 더 실용적인 응용 분야를 가짐(ex. 콜센터, AI 스피커 등)
- closed-set identification과 달리 open-set verification은 훈련에서 보지 못했던 화자의 identity를 확인하는 것을 목표로 하기 때문에, speaker verification은 음성이 discriminative한 embedding 차원의 표현으로 mapping되어야하는 metric learning 문제
- 다른 논문들에서 주로 classification loss를 사용하여 embedding을 학습하였으나 embedding similarity를 최적화하도록 설계되지 않음
- 최근 연구들에서 class 간 분리를 강화하기 위해 verification의 성능을 향상시키는 것으로 알려진 margin variant를 추가한 softmax를 접목시킴 (AM-softmax)

<br/>

- **open-set verification**은 network가 제한된 example을 갖으면서 unseen class에 대해 인식해야하므로 **few-shot learning** 문제라고 볼 수 있음
- few-shot learning 시나리오를 모방하는 **prototypical network**가 제안되었으며, **최근 speaker verification에서 좋은 성능을 달성**하는 것으로 나타남

<br/>

- similarity metric을 최적화하도록 network를 훈련시키기 위해서는 frame-level의 representation(feature)를 utterance-level로 모아야 함
- 가장 단순한 방법은 frame-level을 평균하는 것(TAP, Temporal Average Pooling), 이때 frame들은 모두 같은 weight를 갖게 됨
- verification에 더 discriminative한 frame에 attention하도록 SAP(Self-Attentive Pooling)방법이 제안
- 그러나 instance-level self-attention은 support set(training set)의 특정 sample이 아닌, 일반적으로(training set의 전체 data를 아우름) discriminative한 feature를 찾음; training dataset의 전체적인 특성이 반영되어 특정 sample에 대해서는 효과적이지 않을 수 있음

<br/>

- CAN(Cross Attention Network): few-shot learning에서 최근 support set의 example들과 관련있고, discriminative한 input image의 부분에 attention함으로써 unseen target class를 기반의 attention을 선택할 수 있도록 제안된 방법
- support set의 한 class(speaker)와 utterance를 비교하기 위한 discriminative한 특성이 다른 class와 비교하기 위해 생성되는 특징과 다를 것, 따라서 이 아이디어를 speaker verification에 적용할 수 있음
- 본 논문에서는 frame-level의 정보를 효과적으로 utterance-level의 embedding으로 모으기 위해 support set의 example을 참조하여 attention을 계산하는 CAP(Cross Attentive Pooling)를 제안  
- **이러한 방식으로 network는 support set의 특정 class에 대한 특정 특징을 제공하는 utterance을 식별하고 집중시킬 수 있음**  
- 이는 사람이 unseen class의 instance를 인식할 때, sample 쌍들의 공통적인 특성을 갖는 특징을 찾는 것과 유사함
- instance-level의 pooling과 달리, 제안된 attention module은 class(prototype) feature와 query feature의 관련성을 모델링하여 verification task에서 pair-wise 특성을 최대한 활용

<br/>

---

<br/>

# **Ⅱ. Methods**  

### **2.1 Few-shot learning framwork**

- Speaker recognition을 위한 embedding을 훈련하기 위해 few-shot learning framework인 prototypical network 사용

<br/>

**Batch formation**

- 각 mini-batch에는 support(training) set $S$와 query(test) set $Q$가 포함
- 서로 다른 화자 N명마다 M개의 발화 포함


<center>

$S = {(x_i, y_i)}^{N \times 1}_{i=1}$  

$Q = {(\tilde{x_i}, \tilde{y_i})}^{N \times (M-1)}_{i=1}$  

</center>

> support set은 각 화자마다 1개의 발화를 사용하고, query set은 나머지 발화($2 \leq i \leq M$)를 사용  
> $y, \tilde{y} \in {1, ..., N}$; class label

<br/>

**Training object**

- support set은 단일 발화 $x$로 구성되어, prototype(centroid)는 각 화자 %y%의 support utterance와 같음
<center><img src="https://user-images.githubusercontent.com/46676700/95678027-a5714b00-0c04-11eb-816d-01da565f1eaa.png" alt="img" style="zoom: 80%;" /></center>

<br/>

- log-softmax function을 사용하는 cross-entropy loss는 같은 speaker의 segment 간 거리는 최소화하면서 다른 speaker 간의 거리는 최대화
<center><img src="https://user-images.githubusercontent.com/46676700/95678054-cb96eb00-0c04-11eb-9eb8-6e1a2c6ccb3a.png" alt="img" style="zoom: 80%;" /></center>

<br/>

- query embedding의 크기와 prototype과 query의 cosine similarity를 distance metric으로 사용 (**Normalized prototypical, NP**)
<center><img src="https://user-images.githubusercontent.com/46676700/95678059-d2bdf900-0c04-11eb-808e-efe28e67875f.png" alt="img" style="zoom: 80%;" /></center>

- kye et al.[16]은 speaker embedding을 보다 discriminative하게 만들기 위해 global classification loss와 함께 <span style="background-color:#d2d8d8">episodic training*</span>을 사용
(few-shot task와 유사한 형태의 훈련 task를 통해 모델 스스로 학습 규칙을 도출할 수 있게 함으로써 일반화 성능을 높일 수 있음 [참조-kakaobrainBlog](https://www.kakaobrain.com/blog/106))
- global classification은 support와 query set 모두에 적용
- softmax classification loss를 통합하여 mini-batch에 있는 class뿐만 아니라 모든 class에 대해 discriminative하도록 embedding을 훈련 가능
- **최종적인 objective function**은 동일한 가중치를 적용한 **NP와 softmax cross-entropy loss의 합**(단순 sum)

<br/>

### **2.2 Instance-wise aggregation**

- 이상적인 utterance-level embedding은 frequency가 아닌 temporal 위치에 따라 달라져야함
- 2D convolutional neural network는 2D activation map을 생성하기 때문에 frequency 축만 모두 연결되는 aggregation layer를 [1]에서 제안
- 따라서 pooling layer에 들어가기 전 1xT feature map 생성

<br/>

**Temporal Average Pooling(TAP)**

- 단순하게 temporal domain에 대해 feature의 평균을 취함
<center><img src="https://user-images.githubusercontent.com/46676700/95678556-70ff8e00-0c08-11eb-8d56-7d26175f42c7.png" alt="img" style="zoom: 80%;" /></center>

<br/>

**Self-Attentive Pooling(SAP)**

- 각 시간에 대한 frame 모두 같은 weight를 갖는 TAP와 달리, utterance-level에 더 많은 정보를 제공하는 frame-level에 attention함
<center><img src="https://user-images.githubusercontent.com/46676700/95678562-7ceb5000-0c08-11eb-90d6-3498d822c878.png" alt="img" style="zoom: 80%;" /></center>

> frame-level 특징 $x_t$가 우선 parameter W와 b를 갖는 MLP의 입력으로 넣어 non-linear하게 projection(hidden representation으로 mapping)

<br/>

- hidden vector $h_t$와 훈련되는 context vector $\mu$ 사이의 유사도를 계산하여 score(hidden feature의 상대적인 중요도)로 사용
- softmax function을 통해 나온 결과를 각 frame의 중요도(attention weight)로 사용
- context vector는 speaker recognition에 중요한 정보를 제공하는 high-level representation으로 볼 수 있음
<center><img src="https://user-images.githubusercontent.com/46676700/95678567-807ed700-0c08-11eb-8766-296995b8de48.png" alt="img" style="zoom: 80%;" /></center>

<br/>

- utterance-level embedding $e$는 frame-level 특징과 frame-level의 attention weight와 가중합하여 얻을 수 있음
<center><img src="https://user-images.githubusercontent.com/46676700/95678569-84aaf480-0c08-11eb-8291-24f23db5b892.png" alt="img" style="zoom: 80%;" /></center>

<br/>

### **2.3 Pair-wise aggregation**

- 기존의 instance-wise aggregation과 달리 본 논문에서는 <span style="background-color:#ffed54">**다른 utterance의 frame feature를 사용하여 frame-level feature를 모으는 방법**</span>을 제안
- training과 testing의 목표를 맞추기 위헤 metric기반의 meta-learning framework인 prototypical network 사용

- 이 framework에서 support와 query set pair를 사용하여 CAP를 훈련
- test 시, support set과 query set은 enrollment와 test utterance에 해당

<br/>

 - query와 support set의 모든 utterance pair에 대해 frame-level representation $s={s_1, s_2,\dots, s_{T_s}}, q={q_1, q_2,\dots, q_{T_q}}$ 추출
- meta-projection layer $g_{\Phi}(·)$를 사용하여 frame-level에서 hidden feature 추출
- non-linear projection을 통해 임의의 frame에 빠르게 적응할 수 있으므로 frame pair의 유사도를 잘 측정할 수 있음
- 이 layer는 MLP와 ReLU activation function으로 구성
<center><img src="https://user-images.githubusercontent.com/46676700/95679512-50d2cd80-0c0e-11eb-846c-fa3f1bfe0bde.png" alt="img" style="zoom: 80%;" /></center>

- meta-projection layer 이후, 모든 frame에 대한 hidden representation인 $S, Q$를 얻을 수 있음

> $ S = {S_i}^{T_s}_{i=1}$  
> $ Q = {Q_i}^{T_q}_{i=1}$  
> $S_i, Q_i$ 는 각각 $g_{\Phi}(s_i), g_{\Phi}(q_i)$

<br/>

**Correlation matrix**

- Correlation matrix(상관행렬) R은 가능한 모든 frame pair에 대한 similarity를 요약

<center><img src="https://user-images.githubusercontent.com/46676700/95679513-55978180-0c0e-11eb-8991-dc7ca123ddc4.png" alt="img" style="zoom: 80%;" /></center>

> $R^Q = (R^S)^T$; 순서만 바뀌기 때문에 $R^S$의 transpose가 $R^Q$  
> $R^S_{1, 1}$; support set의 1번째 frame hidden representation과 query set의 1번째 frame hidden representation의 similarity  
> 따라서 $R^S \in \mathbb{R}^{T_s \times T_q}$; [support set frame 수 x query set frame 수]

<br/>

**Pair-adaptive attention**

- pair-adaptive context vector를 얻기 위해 다음과 같이 time축에 대해 correlation matrix를 평균
<center><img src="https://user-images.githubusercontent.com/46676700/95679520-5a5c3580-0c0e-11eb-9c92-c802e2e3bcd0.png" alt="img" style="zoom: 80%;" /></center>

> **$\mu_s \in \mathbb{R}^{T_q}$** 이고, $\mathbb{R}^S_{i,*}$은 $i$번째 row vector

- 논문에서 $T_s$로 되어있는데, $T_s$가 아닌 $T_q$이 되어야 수식적으로 맞는 것 같음 (그림에서는 context vector의 size를 $T_q$로 표기)
- 각 row vector는 다른 utterance의 모든 frame과의 유사도 정보가 있음
- 따라서 다른 utterance의 각 frame에 대한 평균 상관관계를 $\mu$로 표시할 수 있고, 이는 다른 utterance와 얼마나 유사한지 계산하기 위해 context vector로 사용

<br/>

- attention weight는 모든 utterance에 대해 다음과 같이 계산
<center><img src="https://user-images.githubusercontent.com/46676700/95679528-60521680-0c0e-11eb-9a06-8745d6fa010b.png" alt="img" style="zoom: 80%;" /></center>

> $\tau$ : temperature scaling (attention distribution의 선명도 조절) - $\tau \rightarrow \infty$이면 동일한 attention weight를 갖음  

<center><img src="https://user-images.githubusercontent.com/46676700/95679531-647e3400-0c0e-11eb-9e6c-b9b0e4c58a0e.png" alt="img" style="zoom: 80%;" /></center>

- Hou et al [22], utterance-level의 특징을 얻기 위해 residual attention mechanism을 사용
- 다른 utterance에 대해서도 동일한 방법으로 utterance-level feature $q$로 $e_q$를 얻을 수 있음

<br/>

**제안하는 방법의 procedure**

<center><img src="https://user-images.githubusercontent.com/46676700/95680543-7a432780-0c15-11eb-80a4-709be1187867.png" alt="img" style="zoom: 80%;" /></center>
<center><img src="https://user-images.githubusercontent.com/46676700/95680550-829b6280-0c15-11eb-93fc-0ac5babd0115.png" alt="img" style="zoom: 80%;" /></center>

<br/>

---

<br/>

# **Ⅲ. Experiments**

**Model architecture**

<img src="https://user-images.githubusercontent.com/46676700/95680589-cb531b80-0c15-11eb-9d17-c3ead5a27fd8.png" alt="img"/>

<br/>

**Results**

<img src="https://user-images.githubusercontent.com/46676700/95680595-d0b06600-0c15-11eb-8d5a-b8b7166ea620.png" alt="img"/>
