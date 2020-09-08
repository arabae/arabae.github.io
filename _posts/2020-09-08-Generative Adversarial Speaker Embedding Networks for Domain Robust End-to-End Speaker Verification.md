---
layout: post
title: "Generative Adversarial Speaker Embedding Networks for Domain Robust End-to-End Speaker Verification  REVIEW"
subtitle: "Gautam Bhattacharya, Joao Monteiro, Jahangir Alam, Patrick Kenny"
tags: [GAN, SpeakerVerification, SpeakerRecognition, research_review, ML, Unsupervised Domain Adaptation, E2E]
author: Ara Bae
comments: True
---

### ▶ Abstract
- GANs를 이용한 domain invariant speaker embedding을 위한 새로운 접근 방식 제안
	\- source data와 target data로 generator가 embedding을 생성
	\- 생성된 embedding이 source인지 target인지 discriminator가 식별

- **이러한 framework를 사용하여 여러 가지 GAN 변형을 훈련하고 화자 검증에 적용**

- Angular Margin loss를 사용하여 End-to-End model 최적화
<center><img src="https://user-images.githubusercontent.com/46676700/92461324-1e474680-f204-11ea-91bc-e748da169035.png" alt="img" style="zoom: 80%;" /></center>

<br/>

---

<br/>
### Ⅰ. Introduction
\- 화자 embedding : 개인의 identity와 관련된 정보를 포함하는 저차원 벡터 표현

<br/>

**Neural Network기반 화자 embedding**

- 음성 인식, 합성 및 source 분리, 화자 검증 적용 등 다양하게 적용

<br/>

**End-to-End system speaker verification**
- 두 개의 음성 파일에서 embedding을 추출한 뒤 embedding 사이의 cosine distance 등을 사용하여 score 계산
- 모델이 견고하기 위해서 일반적으로 거리 측정 기준을 직접 최적화해야 함 (End-to-End)
- 그러나, 화자 검증에서 훈련하기 어려운 것으로 판단

<br/>

**I-vector system과 동일하게 사용**

- 차원 감소에는 LDA(Linear Discriminant Analysis) 사용
- 검증 시 PLDA(Probabilistic Linear Discriminant Analysis) 사용

<br/>

**NIST SRE 2016 dataset 사용**
- 훈련 데이터(영어)와 테스트 데이터(광둥어 및 타갈로그어) 사이의 mismatch를 도입 (Domain or Covariate shift)
- domain 보상을 위한 적은 양의 label이 없는 target 데이터 제공

<br/>

** 본 논문 저자의 최근 연구에서, End-to-End의 cosine score를 사용하는 domain adversarial 훈련을 이용한 domain 불변 화자 embedding 훈련 제안 (Domain Adversarial Neural Speaker Embeddings, DANSE)**
- Gradient reversal을 사용하여 domain 불변성 및 adversarial grame의 최소화 목표를 달성

<span style="background-color:#f4d451">**본 논문에서는 GANs를 사용하여 unsupervised domain adaptation/invariant로 이전 연구 확장**</span>
< 장점>
- gradient reversal보다 불변성 mapping을 학습하는데 더 나은 gradients 제공
- GAN framework는 gradient reversal보다 더 일반적이고 확장 가능

<br/>

**다양한 GAN 변형**
- 특징 공간의 다른 변형을 생성
- 이러한 특징 공간을 결합이 성능 향상을 가져옴
- Auxiliary Classifier GAN(AuxGAN)의 수정을 제안
- GAN 모델이 DNASE 모델의 성능을 능가
- 다양한 GAN 모델의 score를 평균함으로써 x-vector의 성능보다 향상됨

<br/>

---


<br/>

### Ⅱ. Domain Adaption with GANs

**GAN**
- Generator : target data를 source data의 domain으로 mapping
- Discriminator : source data와 target data의 domain을 구별

<center><img src="https://user-images.githubusercontent.com/46676700/92464311-ecd07a00-f207-11ea-8527-64991f1f261d.png" alt="img" style="zoom: 70%;" /></center>

- 여러 GAN 변형에 해당하는 다른 discriminator의 구성이 특징 공간의 다른 변환을 가져온다는 것을 발견
- vanilla GAN에서 discriminator는 binary cross-entropy(BCE) loss를 최적화하여 훈련

<br/>

**GAN game (기존 GAN loss)**

<center><img src="https://user-images.githubusercontent.com/46676700/92464831-af202100-f208-11ea-9c86-bb4318bebe00.png" alt="img" style="zoom: 50%;" /></center>

- E, D : Embedding(generator), Discriminator 함수

$$
X_s : source\; data\\
X_t : target\; data
$$

<br/>

**Gradients reversal model**
$$
L_{advE} = -L_{advD}
$$

<br/>

---

<br/>

### Ⅲ. Generative Adversarial Speaker Embedding Networks

**본 논문의 목표**

- 화자 embedding model이 특징 추출기(generator)와 domain 식별자(discriminator) 사이의 GAN game을 통해 domain 불변적 특징을 학습
- GAN이 domain 불변성을 갖으며, embedding이 화자를 구분할 수 있어야 함

<br/>

**Loss function (AM-softmax/GAN loss)**

- class간 cosine similarity를 직접 최적화
<center><img src="https://user-images.githubusercontent.com/46676700/92466967-d7f5e580-f20b-11ea-9b8b-ae4db11acd0b.png" alt="img" style="zoom: 50%;" /></center>

- C, E : Classifier, Embedding(generator)  함수

<br/>

<center><img src="https://user-images.githubusercontent.com/46676700/92467164-17243680-f20c-11ea-83c2-adb068c4d9df.png" alt="img" style="zoom: 40%;" /></center>

- s, m : scale factor, margin

<br/>

- BCE loss를 사용하여 domain discriminator를 훈련
<center><img src="https://user-images.githubusercontent.com/46676700/92467390-7d10be00-f20c-11ea-9136-515c58c834d9.png" alt="img" style="zoom: 50%;" /></center>

- 마지막으로, 아래의 loss를 사용하여 discriminator를 속이기 위해 generator(embedding) 훈련
<center><img src="https://user-images.githubusercontent.com/46676700/92467437-9580d880-f20c-11ea-9aa0-c336bf2bd007.png" alt="img" style="zoom: 50%;" /></center>

- embedding 함수는 task loss와 함께 그 다음 adversarial loss 총 2번 학습

<br/>
<br/>

##### 3.1. Auxiliary Classifier GAN

** AuxGAN(ACGAN)**

- 조건(conditional) 이미지 생성을 위해 보조(Auxiliary) loss를 사용하여 GAN을 보완

- side 정보(class label 등)을 예측하는 것이 목표

- D (discriminator) : 2개의 classifier
   \- 데이터가 진짜(real) 인지 가짜(fake) 인지 판별
   \- 해당 데이터의 범주(category)를 분류
   
- G (generator) : label정보와 z(noise)로 가짜 데이터 생성

<center><img src="https://user-images.githubusercontent.com/46676700/92468316-ec3ae200-f20d-11ea-882d-0045ffc0cd5c.png" alt="img" style="zoom: 50%;" /></center>

**원래 ACGAN의 object fuction**

- source의 log-likelihood L_s, class의 log-likelihood L_c
- L_s : 기존 GAN의 목적 함수와 같음 (real/fake 판별)
- L_c : 해당 데이터의 class를 판단 (conditional-GAN, CGAN과 유사)

<br/>

- D(discriminator)는 L_s + L_c를 최대화
- G(generator)는 L_c - L_s를 최대화
$$
L_s = E[logP(s=real|X_{real})] + E[logP(S=fake|X_{fake})]\\
L_c = E[logP(C=c|X_{real})] + E[logP(C=c|X_{fake})]
$$

<br/>

**논문에서 사용한 ACGAN의 object function**

<center><img src="https://user-images.githubusercontent.com/46676700/92469327-9a935700-f20f-11ea-8183-b78231d799d4.png" alt="img" style="zoom: 50%;" /></center>

<br/>

##### 3.2. GAN Variants

**다양한 GAN의 변형 사용**

- 표준 GAN
- Least-Squares GAN
- Relativistic GAN

**각 변형이 특징 공간을 다른 방식으로 변형**

- 모든 모델은 거의 비슷한 성능을 보임

**모든 GAN 모델의 성능을 결합**

- 평균 점수(cosine distance score)를 결합한 것이 최고의 성능을 보임

<br/>

---

<br/>

### Ⅳ.  Experiments and Results

**Training data(source)**

<br/>

- 제안한 DANSE 모델과 x vector, i vector 의 baseline 을 훈련하기 위해 NIST SRE 2004 2010 및 Switchboard Cellular audio 사용 
- 잡음 및 잔향으로 데이터 증강 (128K의 noisy data추가하여, 220K개 사용)
- Adversarial 모델을 훈련시키기 위해 , 5 개 이하의 발화인 화자는 걸러내고 약 6000 명의 화자를 사용
- x-vector, i-vector 는 Kaldi toolkit 사용
- 대부분이 영어 사용자 이며 , 전화를 통해 녹음

<br/>

**Model**

<br/>

- Embedding(generator) 함수는 3X 2 3 input 의 Convolutional layer, 4 개의 residual block, attentive statistics layer, 2 개의 fully connected layer (512, 512) 로 구성
- Classifier는 fully connected layer (64) 와 AM softmax output layer 로 구성 (fully connected layer 가 최종 domain 불편 화자 embedding)
- Discriminator는 2 개의 fully connected layer (256, 256) 와 binary cross entropy output layer 로 구성 
- ELU(Exponential Linear Units)를 모든 계층에 사용
- Batch normalization은 attentive statistics layer 를 사용한 계층에 사용
- AMsoftmax loss 의 s 와 m parameter 는 각각 30 과 0.6 으로 설정

<br/>

**Optimization**

<br/>

- cross entropy 훈련을 사용하여 embedding 특징을 사전 훈련
- 세 가지 네트워크 (embedding 특징 , Classifier, 를 서로 다른 optimizer 사용
- Discriminator는 lr = 0.003 의 RMSprop , Classifier 와 embedding 은lr 0.001 의 SGD 사용

<br/>

**Data sampling**

<br/>

- 훈련 중 훈련 set 의 각 녹음에서 무작위로 audio chunk sampling
- 각 음성을 10 번 sampling (epoch)
- Source data의 mini batch 에 대해 GAN 훈련을 위한 label 이 없는 adaption data 도 동일하게 무작위로 mini batch 를 sampling

<br/>

**Speaker Verification**

<br/>

- Test시 embedding 추출에 필요하지 않은 domain discriminator 를 없앰 
- 64차원의 마지막 hidden layer 가 최종 화자 embedding
- Verification실험은 cosine distance 를 사용하여 score 계산
- 성능의 지표는 EER 사용

<br/>

**Model block**

<br/>
<center><img src="https://user-images.githubusercontent.com/46676700/92470103-e5fa3500-f210-11ea-8ca4-58b5d1bcf508.png" alt="img" style="zoom: 50%;" /></center>
<center><img src="https://user-images.githubusercontent.com/46676700/92470119-ebf01600-f210-11ea-8d0b-bab531d6d72d.png" alt="img" style="zoom: 50%;" /></center>

<br/>

**제안한 adversarial 화자 embedding과 baseline system 성능 비교**

- Baseline시스템 중에서는 DNN 기반의 x vector 시스템이 LDA 차원 감소 추가하는 것 만으로도 i-vector 의 성능보다 향상
- 모든 GAN 기반의 모델이 DANSE 보다 더 나은 성능을 보임
- AuxGAN(ACGAN), LSGAN, RelGAN embedding 의 score 를 평균한 것이 가장 성능을 크게 개선함

<br/>

<center><img src="https://user-images.githubusercontent.com/46676700/92470359-3d98a080-f211-11ea-8d38-75adaeb55df0.png" alt="img" style="zoom: 50%;" /></center>

<br/>

---

<br/>

### Ⅴ.  Conclusion

- GANs를 이용한 domain 불변 화자 embedding 학습을 위한 새로운 framework 제안
- 여러 가지 GAN 의 변형을 학습하여 score 를 결합함으로써 크게 향상된 성능을 얻음
- End-to-End model 에 최적화되어 있으며 간단한 cosine distance 를 사용하여 score 를 계산

<br/>

- 향후 특징 공간과 데이터 공간 GAN 의 결합 및 GAN 기반 특징 공간 증강 방법과 같이 다른 adversarial 전략을 고려할 것

