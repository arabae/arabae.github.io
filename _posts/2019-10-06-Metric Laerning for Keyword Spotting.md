---
layout: post
title: "Metric Learning for Keyword Spotting"
date: 2019-10-06
category: review
thumbnail: /style/image/metric_keyword.jpg
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Jaesung Huh, Minjae Lee, Heesoo Heo, Seongkyu Mun, Joon Son Chung</span>

# 📌 **Abstract**

- **목표 : Metric learning을 통해 keyword spotting(음성 입력 중에 특정 단어를 발화하였는지 검출)을 위한 효과적인 representations를 훈련하는 것**

- **기존 방법** : target/non-target keyword들이 모두 사전에 정의된 closed-set classification 문제만 다루기때문에 unseen non-target에 대해 성능이 저하되어 real-world에서 높은 FAR(False Alarm Rate)을 보임  

- keyword spotting은 다양한 unknown sound에서 사전에 정의된 target keyword를 detection하는 문제로, unseen/unknown non-target이 target keyworkd와 명확히 구별되어야 한다는 점이 metric learning과 유사한 점이 많음

- 주요 차이점은 **target keyword가 알려져 있고, 이미 정의되어 있다는 점**

- <span style="background-color:#ffed54">따라서 **target keyword와 non-target keyword 사이의 거리를 최대화** 하고 분류 목표에 따라 **target keyword에 대한 class 별 가중치를 학습** 하는 metric learning기반의 새로운 방법을 제안</span>

- Goodle Speech Commands datasest으로 실험을 수행하였으며 전체적인 classification 정확도를 유지하면서 unseen non-target keyword들에 대한 FA(False Alarm)을 크게 감소시킴

<br/>

---

<br/>


# **Ⅰ. Introduction**

**✔ Keyword Spotting(KWS)**

- 다양한 mobile device를 동작시키는 단어(wake-up words, "OK Google", "Hey Siri", "Alexa") 또는 자주 사용하는 짧은 명령과 같이 미리 정의된 작은 음성 신호 집합을 detection하는 작업
- 최근 CNN기반의 architecture들이 이 분야에서 좋은 성능을 달성하였고, 주로 target keyword와 일반적인 음성이나 잡음같은 non-target sound를 구분하는 classifier를 기반함
- non-target class는 매우 다양할 수 있지만 이전 작업들에서는 제한된 수의 non-target class만 사용하여 실제 환경을 충분히 반영하지 못하였음 (전통적인 방법은 후처리를 통해 FA을 줄이려했지만 deep learning 접근법과 함께 사용되지 않았음)

<br/>

- 실제 keyword spotting은 사전에 정의된 keyword가 알려지지 않은 다양한 소리에서 발견되는 classification 문제가 아닌 detection 문제와 유사하지만 이전의 많은 작업에서는 non-target sound를 단일 class로 간주하였음
- 본 논문에서는 **target 발화를 accept하거나 reject**하는데 사용할 수 있는 **discriminative한 embedding을 학습**하는 **metric learning**에서 영감을 받았으며 화자검증과 유사하지만 keyword가 미리 정의되어 있다는 점이 다름
- metric learning 방법은 input signal을 embedding 공간에 mapping하여 class 간 분산을 크게하고, class 내의 분산은 작게 함 ("contrastive loss" - face recognition, speaker verification에서 주로 사용하는 방법)

<br/>

- 최근 metric learning 기술은 constrastive, triplet loss의 단점(pair 선택의 어려움)을 극복하기 위해 도입
- [17, 18]논문에서 훈련 중 여러개의 positive와 negative를 사용해 careful pair 선택이 필요하지 않은 훈련 방법을 제안하였으며, Siamese neural network(dynamic time wraping기반 speech recognition)에 사용되는 frame-wise embeding을 훈련하는 화자 검증에서 성능 향상을 보임

<br/>

**✔ metric learning에서 영감을 받은 keyword spotting을 위한 여러 방법을 제안**

-  Network architecture 를 유지하면서 loss functions을 classification에서 다양한 metric learning으로 변경
-  target class 내 거리를 최소화하기 위해 embedding을 훈련, non-target embedding과 거리는 사용하지 않음(실제 keyword spotting에서는 이 부분은 다루지 않기 때문)
-  잠재적으로 무한한 non-target sound와 유사성을 비교하여 사용하는 classification과 대조적인 방법

<br/>

**✔ 기여한 바**

- Google Speech Command Dataset사용, 제안하는 방법이 classification task에 대해 정확성은 유지하면서 detection에 대해 classification 기반 baseline system들 보다 우수한 성능을 보임
- 1) keyword spotting이 이전 task와 달리 detection의 문제 중 하나로 정의
- 2) non-target ketword의 정확도를 크게 높일 수 있는 mectirc learning 기반 방법 제안


<br/>

---

<br/>


# **Ⅱ. Metric Learning Framework**

- metric learning에 사용되는 기존 loss fuction에 대해 설명하고, 전체적인 classification 정확도를 유지하면서 non-target의 정확도도 높이기 위한 수정된 방법을 제안


### <span style="background-color:#aee4ff">**2.1 Loss functions**</span>  

**Triplet loss**

- 가장 일반적인 ranking loss fuction 중 하나
- 동일한 class의 embedding 사이 거리가 줄어들고, 동시에 다른 class의 embedding과는 거리가 멀어지게 학습됨
- $f(x;w) ∈ R^D$ : input을 embedding 공간으로 mapping하는 함수라고 가정

<center><img src="https://user-images.githubusercontent.com/46676700/95653158-5a3a3800-0b31-11eb-94ce-6f077868a0f7.png" alt="img" style="zoom: 50%;" /></center>


> $x_i, {x'}_i$ : 같은 class $i$으로부터 얻은 input samples  
> $x_j$ : 다른 class $j(j{\neq}i)$로부터 얻은 sample  
> $\|{x-y}\|$ : $x$와 $y$간 pairwise-distance

- triplet $P_{i,j} = (x_i, {x'}_i, x_j)$일때, triplet loss L은 batch에 대해 minimized되어 훈련
> 여기서 $\alpha$는 constant margin (e.g. $\alpha=1$)

- $\|f(x_i)-f({x'}_j)\| < \|f(x_i)-f(x_j)\| + \alpha$; "같은 class에서 나온 sample들의 거리가 다른 class의 sample보다 가까울 것이다." 에서 발전된 loss

<br/>

**Prototypical networks**

- open classification을 수행하기 위한 mectirc space를 학습하기 위해 제안
- 각 class의 prototype representations(embedding) 간의 distance를 계산
- GE2E loss의 distance metric을 변형한 prototypical loss의 angular 변형을 실험에 사용
- 각 mini-batch는 서로 다른 class N개당 M개의 발화가 있는 NxM을 input feature로 사용

<center><img src="https://user-images.githubusercontent.com/46676700/95653587-b0f54100-0b34-11eb-81a3-9df0ff078997.png" alt="img" style="zoom: 50%;" /></center>

> $e_{j,M}$ : 각 batch에서 class $j$의 query(embedding)  
> $c_k$ : class k의 centroid ($S$에서 target이 되는 utterance index(M)는 제외한 embedding의 평균)  

- cosine을 기반의 Similarity metric을 사용하는 angular prototypical loss는 L2 distance보다 stable(안정적)하고, robust(강인함)

> learnable parameter $w > 0$와 $b$를 사용  

- 각 batch에서 angular prototypical loss의 목적은 해당 embedding과 같은 class의 centroid와 유사성은 최대화하면서 다른 class의 centroid와는 최소화하는 것이므로 다음과 같이 정의하여 사용

<center><img src="https://user-images.githubusercontent.com/46676700/95653779-3a594300-0b36-11eb-93b0-d7202056d818.png" alt="img" style="zoom: 50%;" /></center>

<br/>

### <span style="background-color:#aee4ff">**2.2 Pair selection strategy**</span>

- 2.1에서 소개한 loss function을 사용하여 network를 훈련시키는 방법 소개
- 'target' keyword와 unknown 'non-target' sound을 효과적으로 구별하기위해 positive, negative pair를 선택하는 방법을 주로 다룸

<br/>

**mectirc learning with an unknown cluster**

- 2.1의 baseline metric learning 접근법을 사용하는 방식
- 이 접근 방법에서는 target keyword와 non-target keyword 모두 triplet 또는 prototypical loss를 사용하여 embedding space의 anchor 또는 centroid를 기반으로 각 class에 맞게 cluster됨
(2개의 target/non-target class로 sample들을 분류함)
- target keyword와 non-target keyword를 단순하게 하나의 class로 취급하지만, non-target keyword의 variance는 매우 높을 수 밖에 없음(target은 특정되었는데 non-target은 매우 다른 sound들의 집합이기 때문에)

<br/>

- 본 논문에서는 classification을 목표로 두고 훈련하지 않기때문에 target/non-target인지 확인하기 위해 새로운 추론 방법을 제안
- <span style="background-color:#ffed54">**network를 훈련 시킨 후, 전체 training data의 embedding을 추출하여 평균으로 centroid를 계산**</span>
- <span style="background-color:#ffed54">**test 단계에서 마찬가지로 embedding을 추출하고 위에서 계산된 centroid들과 유사성을 계산하여 어떤 class에 속하는지 결정**</span>
- 실제 시나리오에서도 target이 이미 알려져 있으므로 각 class에 해당하는 centroid를 미리 계산하여 모델의 parameter로 사용할 수 있어, input이 주어졌을 때 훈련된 모델에서 embedding을 얻은 뒤 각 centroid와 거리를 계산해 어떤 class에 속하는지 분류할 수 있음

<br/>

**Metric learning without an unknown cluster**

- non-target keyword는 target keyword들을 제외한 모든 소리와 음성을 포함하기 때문에 범위가 훨씬 큼
- 그러나 기존의 접근 방식에서는 non-target keyword를 하나의 단일 class로 두 때문에 다양한 non-target embedding들이 이 단일 class에 맞도록 훈련됨 (variance를 고려하지 않음)
- 제한된 non-target keyword로 unseen word들을 일반화하는 것은 어렵기 때문에, **학습 중 non-target keyword를 하나의 point(class)로 clustering하지 않도록 수정을 제안**

<center><img src="https://user-images.githubusercontent.com/46676700/95654278-f2d4b600-0b39-11eb-9905-5a6b7c6c65a8.png" alt="img" style="zoom: 50%;" /></center>

- non-target인지 구별할 때, centroid를 사용할 수 없기 때문에(모든 sample들을 알 수 없어서) 추가적인 단계가 필요
- metric learning으로 embedding extractor를 훈련한 뒤, training data의 embedding으로 1대 나머지를 구분할 수 있도록 RBF(Radial Basis Function) kernel SVM을 훈련
- 이 SVM을 사용하여 test set의 class를 결정

<br/>

### <span style="background-color:#aee4ff">**2.3 Prototypical networks with fixed target classes**</span>

- target keyword spotting을 위한 수정된 prototypical loss 제안
- prototypical network의 원래 framework에서 centroid는 few-shot learning setting의 inference동안 계산됨
- 그러나 얼굴 및 화자 검증과 같은 prototypical network와 달리 target keyword가 고정되어 있다는 사실을 활용할 수 있음
- 따라서 알려진 keyword의 경우 즉석에서 계산되는 각 class의 중심 $c_k$를 학습되는 class별 가중치 $W_k$로 대체

<br/>

- 실험에 따르면 classifier기반의 keyword spotting system은  mectirc learning기반 system보다 target keyword에 대해 더 높은 정확도를 갖는 반면, non-target keyword에 대한 FAR이 더 낮았음
- 제안된 방법은 학슬된 class별 weight를 사용하여 알려진 keyword를 감지함으로써 두 방법의 장점을 통합하는 동시, mectirc learning과 유사한 방식으로 non-target을 reject할 수 있음

<br/>

<span style="background-color:#ffed54">**AP-FC(Angular Prototypical with Fixed Classes); 제안하는 방법**</span>

<center><img src="https://user-images.githubusercontent.com/46676700/95655613-9f676580-0b43-11eb-8598-233384fd1329.png" alt="img" style="zoom: 50%;" /></center>

> $S_{i,j,k}$ : class $j$의 $i$번째 embedding과 centroid대신 사용되는 $k$번째 target keyword의 훈련되는 parameter $W_k$ 사이의 scaled cosine similarity  
> $W_k$ : target keyword에 대해 학습되는 유일한 parameter  
> classifier에서 output layer의 역할을 함($W_k$와 계산해서 나온 결과가 class 분류에 사용되므로)  

<center><img src="https://user-images.githubusercontent.com/46676700/95655621-a7270a00-0b43-11eb-9ece-3afd4ec1d84f.png" alt="img" style="zoom: 50%;" /></center>

> $N'$ : 하나의 mini-batch에 포함된 non-target keyword의 sample 수  
> eq.8에서 non-target의 class index는 N이라고 가정  
> 실험에서 $N'$의 값을 6으로 설정  

- 모든 $k ∈ {target}$에 대한 학습되는 parameter $W_k$는 각 target keyword의 centroid 역할을 하도록 훈련되었을 것이라 기대
- 하나의 mini-batch에는 target과 non-target의 균형을 조정하기 위해 각 target keyword에 대한 하나의 sample과 non-target keyword의 여러 sample을 포함
- L을 최소화하면 분자에 있는 값(k번째 class의 embedding과 parameter의 거리)이 작아지므로 해당 class에 속한 embedding과 parameter가 점점 가까워질 것
