---
layout: post
title: "Attentive Statistics Pooling for Deep Speaker Embedding"
date: 2019-05-01
category: review
thumbnail: /style/image/attention.png
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Koji Okabe, Takafumi Koshinaka, Koichi Shinoda</span>

# 📌 **Abstract**
- <span style="background-color:#FFE49B">**Text-independent**(문장 독립 : 발화 내용이 동일하지 하지 않음)한 **Speaker Verification**(화자 검증 : 등록된 화자인지 아닌지 판단, SV)에서 **Deep speaker embedding을 위한 attentive statistics pooling** 제안</span>

- 기존의 speaker embedding에서는 단일 발화의 모든 frame에서 frame-level의 특징을 모두 평균 내어 utterance-level의 특징을 형성

- 제안하는 방법은 attention mechanism을 사용하여 각 frame마다 다른 weight(가중치)를 부여하고, weighted mean(가중 평균)과 weighted standard deviations(가중 표준 편차)를 생성


✔  <span style="background-color:#FFE49B">NISE SRE 2012 및 VoxCeleb data set에서 기존 방법에 비해 EER이 각각 7.5%, 8.1% 감소</span>

<br/>

---

<br/>

# 📌 **Introduction**

- **화자 인식은 지난 10년동안 i-vector paradigm과 진화**하였고, i-vector는 고정된 저차원의 특징 벡터 형태로 음성 발화 혹은 화자를 표현

- 다양한 기계학습을 통해 Deep learning이 성능 향상에 크게 기여하며, 화자 인식을 위한 특징 추출에 Deep learning을 도입이 증가

- 초기 연구에서는 ASR(Automatic Speech Recognition)의 음향 모델에서 도출된 DNN을 UBM으로 사용하여 기존의 GMM기반 UBM보다 우수한 성능을 보였지만 언어 의존성 단점과 훈련을 위해 음소 transcription이 필요

- 최근 **DNN은 이러한 i-vector framework와 독립적**으로 **화자 마다 고유한 특징 벡터를 추출하는데 유용**하다고 밝혀짐 (특히, 짧은 발화 조건에서 더 나은 성능을 보임)

- Text-dependent(문장 종속 : 발화 내용이 동일함) SV에서 LSTM(마지막 frame에서 하나의 출력을 갖는 구조)을 사용하여 utterance-level의 특징을 얻는 End-to-End Neural Network기반의 방법이 제안되었으며, 기존의 i-vector보다 좋은 성능을 보임

- Text-independent SV는 입력으로 다양한 길이의 발화를 갖으므로 average pooling layer가 도입되어 frame-level의 화자 특징 벡터를 일정한차원을 갖는 speaker embedding 벡터를 얻음

- 대부분 최근 연구에서 DNN이 i-vector보다 더 나은 정확도를 갖는 것을 보여주며 Snyder 외는 average pooling를 확장한 statistics pooling (평균 및 표준 편차 계산)을 채택

- 그러나 아직 정확도 향상에 대한 표준 편차 pooling의 효율성은 보고하지 않음

<br/>

- 최근 다른 연구에서는 이전에 기계 번역에서 상당한 성능 향상을 가져온 **attention mechanism과 통합**

- 화자 인식에서도 중요도 계산 시, speaker embedding 추출하는 network의 일부로 작동하는 작은 attention network 사용

- 계산된 중요도는 frame-level의 특징 벡터의 weighted mean 계산할 때 사용하여 speaker embedding이 중요한 frame에 초점을 맞춤

- 그러나 이전 연구에서는 고정 길이의 text-independent 혹은 text-dependent 화자 인식과 같은 제한된 작업에서만 수행

**- 본 논문에서 attention mechanism으로 계산된 중요도로 importance-weighted standard deviation과 weighted mean사용한 새로운 pooling방법인 attentive statistics pooling를 제안**

- 가변 길이의 text-independent한 환경에서 attentive statisitics pooling을 사용하는 첫 번째 시도 이며, 다양한 pooling layer 비교를 통해 표준 편차가 화자 특성에 미치는 효과를 실험적으로 보여줌

<br/>

---

<br/>

# 📌 **Deep speaker embedding**

- 기존의 DNN을 사용한 speaker embedding 추출 방법

> input : acoustic feature (MFCC, filter-bank 등)  
> frame-level의 특징 추출을 위해 TDNN, CNN, LSTM 등의 Neural Network  
> 가변 길이의 frame-level 특징을 고정 차원의 벡터로 변환하기 위한 pooling layer  
> utterance-level의 특징을 추출하기 위한 fully-connected layer(hidden layer 중 하나의 node 수를 작게 하여 bottleneck feature로 사용)  

<br/>

<center><img src="https://user-images.githubusercontent.com/46676700/89165519-a443f200-d5b3-11ea-8009-d34a68859aa4.png" alt="img" style="zoom:60%;" /></center>

<br/>

---

<br/>

# 📌 **High-order pooling with attention**

< Statistics pooling - 기존에 사용하던 pooling 방법 >

- frame-level 특징에 대해 평균(mean)과 표준 편차(standard deviation) 계산 (⊙ : Hadamard 곱)하여 concatenation

<center><img src="https://user-images.githubusercontent.com/46676700/89165568-b160e100-d5b3-11ea-9a93-2a31b6530b2b.png" alt="img" style="zoom: 45%;" /></center>

< Attention mechanism >

- 기계 번역에서 긴 문장의 성능 저하를 해결하기 위해 모델이 출력 단어를 예측할 때 **특정 단어를 집중**해서 보는 방법을 도입

<center><img src="https://user-images.githubusercontent.com/46676700/89165571-b1f97780-d5b3-11ea-91e3-8fa3f49000fc.png" alt="img" style="zoom: 80%;" /><img src="https://user-images.githubusercontent.com/46676700/89165573-b1f97780-d5b3-11ea-9545-3a591f97f98d.png" alt="img" style="zoom: 50%;" /></center>

<br/>

<img src="https://user-images.githubusercontent.com/46676700/89165553-aefe8700-d5b3-11ea-9e0a-c4c8d5fc14a0.png" alt="img"/>

- decoder의 <span style="color:#a5cbf0">**시간 i(현재)에서 hidden state 벡터**</span>는 <span style="color:#a5cbf0">**시간 i-1(이전)의 hidden state 벡터**</span>와 <span style="color:#ffaddf">**시간 i-1(이전)에서 decoder의 output**</span>, 그리고 <span style="color:#7cbfb6">**시간 i(현재)에서의 context 벡터**</span>를 입력으로 계산

<img src="https://user-images.githubusercontent.com/46676700/89165558-af971d80-d5b3-11ea-84c7-8f0478e8e680.png" alt="img"/>

- <span style="color:#7cbfb6">**context 벡터**</span>는 **시간 i에서 입력 x에 대한 길이 T** 전체에 대한 **<span style="color:#f9d877">encoder hidden state 벡터</span>**의 **가중합**으로 계산

<img src="https://user-images.githubusercontent.com/46676700/89165559-b02fb400-d5b3-11ea-9ad9-a8383a6810d6.png" alt="img"/>


- <span style="color:#33558c">**시간 i에서 j번째 단어의 energy**</span>는 **<span style="color:#a5cbf0">시간 i-1(이전)에서 decoder hidden state</span>**와**<span style="color:#f9d877"> j번째 encoder hidden state</span>**가 입력인 **aligment model(a)** 결과값 (alignment model은 tanh, ReLU 등 activation function)

<img src="https://user-images.githubusercontent.com/46676700/89165560-b02fb400-d5b3-11ea-8753-68026664a442.png" alt="img"/>

<br/>

< Attentive statistics pooling >

<center><img src="https://user-images.githubusercontent.com/46676700/89165563-b0c84a80-d5b3-11ea-9590-62c129a447e4.png" alt="img" style="zoom: 50%;" /><img src="https://user-images.githubusercontent.com/46676700/89165564-b0c84a80-d5b3-11ea-8a2f-c887055c76d8.png"  alt="img" style="zoom: 50%;" /></center>

attention mechanism을 사용하여 계산한 **가중치를 통해 mean과 standard deviation을 갱신**

<center><img src="https://user-images.githubusercontent.com/46676700/89165566-b160e100-d5b3-11ea-9625-41ccb0db4353.png"  alt="img" style="zoom: 67%;"/></center>

<br/>

---

<br/>

# 📌 **Experimental settings**

**i-vector**

> input : 60차원 MFCC  
> UBM : 2048 mixture  
> TV matrix, i-vector : 400차원  
> Similarity score : PLDA  

<br/>

**Deep speaker embedding**

> input : 20차원(SRE 12), 40차원(VoxCeleb) MFCC  
> hidden layer : 5-layer TDNN(activation function : ReLU, node : 512)  
> pooling dimension : 1500차원  
> acoustic feature vector(MFCC) 15개 frame으로 frame-level 특징 생성  
> 2 fully-connected layer (1st : bottleneck feature - 512, activation function : ReLU, batch   normalization)  
