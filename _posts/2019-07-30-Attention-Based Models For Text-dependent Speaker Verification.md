---
layout: post
title: "Attention-based Models For Text-dependent Speaker Verification : REVIEW"
subtitle: "F A Rezaur Rahman Chowdhury, Quan Wang, Ignacio Lopez Moreno, Li Wan"
tags: [Attention, pooling, LSTM,, SpeakerVerification, SpeakerRecognition, research_review]
author: Ara Bae
comments: True
---

### ✏ Abstract 🔎
- Attention 기반 모델 : 입력 sequence의 전체 길이를 요약할 수 있는 능력
- 음성 인식, 기계 번역, 이미지 캡션과 같은 다양한 곳에서 뛰어난 성능을 보임
- End-to-End Text-dependent 화자 인식 시스템에서 attention mechanism 사용을 분석
- 다양한 attention layer의 변형을 연구하고 attention weight에 대한 다양한 pooling방법을 비교
- Attention mechanism을 사용하지 않은 LSTM과 성능 비교

<br/>

---

<br/>
### Ⅰ. Introduction 🔎

**✔ Global Password Text-dependent Speaker Verification(SV) 시스템**

- 등록 및 테스트 발화가 특정 단어로 제한 (Text-dependent)
- “Ok-Google”과 “Hey Google” 사용 ( Global password)


<br/>

**✔ 현재 가장 많이 접근하고 있는 훈련 방법**

- 등록 및 테스트하는 단계를 시뮬레이션하는 End-to-End 구조
- [6]논문 “i-vector+PLDA 시스템을 그대로 모방한 구조”의 경우, 더 나은 성능을 위해 모델을 규제하였으나 초기화를 위해 기존의 i-vector와 PLDA 모델이 필요
- [7] 논문, TD-SV task에서 LSTM 네트워크가 기존 End-to-End DNN보다 더 나은 성능을 보여줌

<br/>

**✔이전 논문에서의 문제점**

- 묵음과 배경 잡음이 많이 없음
- 본 논문에서는 keyword 검출에 의해 분할된 800ms의 짧은 frame이지만, 묵음과 잡음이 있음

<br/>

**✔이상적인 Embedding 생성**

- 음소에 해당하는 frame을 사용하여 제작
- 입력 sequence 중 관련성이 높은 요소를 강조하기 위해 attention layer 사용

<br/>

---


<br/>

### Ⅱ. Baseline Architecture🔎

#### <span style="background-color:#aee4ff">**TE2E model**</span>

**✔  baseline end-to-end training architecture**

<center><img src="https://user-images.githubusercontent.com/46676700/94424981-1573e000-01c6-11eb-8bf5-4890542a60db.png" alt="img" style="zoom: 80%;" /></center>

- 훈련 단계에서, 하나의 평가용 발화 𝒙𝑗~와 N개의 등록 발화 𝒙𝑘𝑛 (𝑓𝑜𝑟 𝑛=1, …, 𝑁) tuple이 LSTM network의 입력으로 사용

> {𝒙𝑗~, (𝒙𝑘1, …, 𝒙𝑘𝑁)} ; input
> 
> 𝒙 : 고정 길이의 log-mel fiterbank feature
> 
> 𝑗, 𝑘 : 발화한 화자 (j와 k는 같을 수 있음)
> 
> 만약 𝒙𝑗~와 𝑀 개의 등록 발화가 같은 화자라면 tuple positive (𝑗=𝑘), 다르면 negative

- ℎ𝑡 : t번째 frame에서 LSTM의 마지막 layer의 출력 ( 고정 차원의 vector )
- 마지막 frame의 output을 d-vector 𝝎 (ℎ𝑇) 로 정의

> {𝝎𝑗~, (𝝎𝑘1, …, 𝝎𝑘𝑁)} ; output
> 
> Tuple (𝝎𝑘1, …, 𝝎𝑘𝑁)을 평균내어 centroid 계산

<br/>

<center><img src="https://user-images.githubusercontent.com/46676700/94425430-e447df80-01c6-11eb-9148-c79bd11b149b.png" alt="img" style="zoom:80%;" /></center>

<br/>

**✔  Cosine Similarity Function 정의**

<center><img src="https://user-images.githubusercontent.com/46676700/94425434-e611a300-01c6-11eb-990c-2a6bc83ad06b.png" alt="img" style="zoom:80%;" /></center>

<br/>

**✔  Loss Function 정의**

<center><img src="https://user-images.githubusercontent.com/46676700/94425445-e7db6680-01c6-11eb-9729-e41c138555a5.png" alt="img" style="zoom: 80%;" /></center>

<br/>


---


<br/>

### Ⅲ. Attention-based Model
#### <span style="background-color:#aee4ff">**3.1 Basic attention layer**</span>

**✔  Baseline system과 차이점**

- 마지막 frame의 출력을 d-vector(𝝎)로 직접 사용
- Attention layer는 각 t frame 에서의 LSTM 출력 ℎ𝑡에 대한 스칼라 점수 𝑒𝑡 를 훈련하여 weighted sum한 결과로 d-vector(𝝎) 정의

<center><img src="https://user-images.githubusercontent.com/46676700/94430186-76071b00-01ce-11eb-8ae9-0fdf5abcf182.png" alt="img" style="zoom: 80%;" /></center>

- Normalized weight 𝛼𝑡와 weighted sum한 결과 d-vector는 다음과 같이 정의 

<center><img src="https://user-images.githubusercontent.com/46676700/94430336-ac449a80-01ce-11eb-8094-4fcf8644fec6.png" alt="img" style="zoom: 80%;" /></center>

<center><img src="https://user-images.githubusercontent.com/46676700/94430342-ae0e5e00-01ce-11eb-9395-90efff7c8674.png" alt="img" style="zoom: 80%;" /></center>

<br/>

- **aritecture로 보는 차이점**
<center><img src="https://user-images.githubusercontent.com/46676700/94430460-eca41880-01ce-11eb-9807-6a7dea6d97fa.png" alt="img" style="zoom: 80%;" /></center>


<br/>

#### <span style="background-color:#aee4ff">**3.2 Scoring functions**</span>

- Bias-only attention
여기서 b𝑡는 scalar. LSTM 출력 h𝑡에 의존하지 않음.

<center><img src="https://user-images.githubusercontent.com/46676700/94430647-34c33b00-01cf-11eb-87f5-e43a51edc41a.png" alt="img" style="zoom: 80%;" /></center>

- Linear attention
여기서 w𝑡는 m차원 vector, b𝑡는 scalar. frame마다 다른 parameter가 사용

<center><img src="https://user-images.githubusercontent.com/46676700/94430651-368cfe80-01cf-11eb-85a2-d759801a1634.png" alt="img" style="zoom: 80%;" /></center>

- Shared-parameter linear attention
모든 frame에 대해 m차원 vector  w와 scalar b가 동일하게 사용

<center><img src="https://user-images.githubusercontent.com/46676700/94430653-37be2b80-01cf-11eb-95d0-af0d4afd142b.png" alt="img" style="zoom: 80%;" /></center>

- Non-linear attention
여기서 𝑾𝒕는 m’ X m matrix, 𝐛𝑡와 𝐯𝑡는 m’차원의 vector(차원 m’은 훈련 데이터 셋에서 조정)

<center><img src="https://user-images.githubusercontent.com/46676700/94430710-50c6dc80-01cf-11eb-8673-5af3e52f4b04.png" alt="img" style="zoom: 80%;" /></center>

- Shared-parameter non-linear attention
모든 프레임에 대해 동일한 parameter 𝐖, 𝐛, 𝐯 를 공유

<center><img src="https://user-images.githubusercontent.com/46676700/94430715-51f80980-01cf-11eb-9b90-9a302bca378a.png" alt="img" style="zoom: 80%;" /></center>

<br/>

#### <span style="background-color:#aee4ff">**3.3 Attention layer variants**</span>

- 기본적인 attention layer와 달리 두가지의 변형된 기법 Cross-layer attention와 Divided-layer attention 소개

**✔ Cross-layer attention**

- 기존의 방법 : 마지막 LSTM의 layer의 출력 h𝑡 (1≤𝑡≤𝑇)를 사용하여 score e𝑡와 weight α𝑡를 계산
- 변형된 방법 : 중간 LSTM layer의 출력 h'𝑡(1≤𝑡≤𝑇)으로 계산 (그림 3.(a) output에서 마지막 2번째 layer를 사용하는 경우)
- d-vector 𝝎는 여전히 마지막 layer 출력 h𝑡와 weighted sum으로 계산

<center><img src="https://user-images.githubusercontent.com/46676700/94431728-9df77e00-01d0-11eb-83a4-7694a369266d.png" alt="img" style="zoom: 80%;" /></center>

<br/>

**✔ Divided-layer attention**

- 마지막 LSTM layer의 출력 h𝑡의 차원을 2배로 늘리고 그 차원을 part a와 part b 두 부분으로 균등하게 나눔
- part b를 사용하여 weight를 계산하고, 나머지 part a와 weighted sum하여 d-vector 생성

<center><img src="https://user-images.githubusercontent.com/46676700/94431901-e1ea8300-01d0-11eb-80d9-464a2cafaf02.png" alt="img" style="zoom: 80%;" /></center>

<br/>

#### <span style="background-color:#aee4ff">**3.4 Weights pooling**</span>

**✔ Basic attention layer의 또 다른 변화**

- LSTM의 output ℎ를 average하기 위해 normalized weight 𝛼𝑡 를 직접 사용하지 않고, maxpooling으로 선택적으로 사용

**✔ 두 가지 maxpooling 방법 사용**

- Sliding Window maxpooling : Sliding window안의 weight 중 큰 값만 두고, 나머지는 0으로 만듦 
- Global top-K maxpooling : 가장 큰 K개의 값만 두고, 나머지는 0으로 만듦

<center><img src="https://user-images.githubusercontent.com/46676700/94432216-63421580-01d1-11eb-8235-ee4f90a727af.png" alt="img" style="zoom: 80%;" /></center>

> t번째 pixel : 가중치 𝛼𝑡
> 
> 밝을 수록 가중치가 큰 값을 의미


---

<br/>

### Ⅳ. Experiments  🔎

#### <span style="background-color:#aee4ff">**4.1 Datasets and basic setup**</span>

**✔  사용한 Dataset**

- “Ok Google”과 “Hey Google”이 혼합된 발화 데이터
- 약 630K 화자가 150M 발화 (테스트 데이터 : 665명 화자)
- 평균적으로 enrollment는 4.5개, evaluation은 10개의 발화로 구성

<br/>

**✔  Basic setup**

- 기본 baseline은 3개의 layer로 이루어진 LSTM
- 각 layer는 128차원이며, 64차원으로 projection하는 linear layer를 가지고 있음
- Global password만 포함하는 길이 T=80 frame(800ms)의 세그먼트로 분리하는 keyword detection 후 40차원의 log-mel-filterbank feature 생성
- MultiReader기법을 사용하여 두 개의 keyword를 혼합하여 사용

<br/>

#### <span style="background-color:#aee4ff">**4.2 Basic attention layer**</span>

- 다양한 점수 계산 함수를 사용하여 Basic attention layer과 비교

<center><img src="https://user-images.githubusercontent.com/46676700/94432416-b4eaa000-01d1-11eb-8141-99d48b30f3da.png" alt="img" style="zoom: 80%;" /></center>

- Bias-only와 linear attention은 EER이 거의 개선되지 않음
- Non-linear 중 특히, shared-parameter의 경우 성능 향상이 있음

#### <span style="background-color:#aee4ff">**4.3 Variants**</span>

- Basic attention layer와 두 가지 변형(cross-layer, divided-layer) 비교
- 이전 실험에서 최고의 성능을 낸 shared-parameter non-linear scoring function을 사용

<center><img src="https://user-images.githubusercontent.com/46676700/94432517-d8ade600-01d1-11eb-8325-50d593324e2b.png" alt="img" style="zoom: 80%;" /></center>

- cross-layer는 마지막에서 2번째 layer에서 score를 훈련 
- divided-layer attention이 마지막 LSTM layer의 차원이 2배이지만, Basic attention과 cross-layer attention보다 약간 더 나은 성능을 보임


#### <span style="background-color:#aee4ff">**4.4 Weights pooling**</span>

- Attention weight를 다양한 pooling방법으로 사용한 것과 비교
- Shared-parameter non-linear scoring function과 divided-layer attention 사용
- Sliding window maxpooling : 10 frame window size와 5 frame step size
- Global top-K maxpooling : K = 5

<center><img src="https://user-images.githubusercontent.com/46676700/94432565-ea8f8900-01d1-11eb-856a-11c004b078e2.png" alt="img" style="zoom: 80%;" /></center>

- Sliding window maxpooling이 EER이 약간 더 낮은 것을 확인

<br/>

**✔ 각 방법에서 attention weight를 visualization**

<center><img src="https://user-images.githubusercontent.com/46676700/94433266-03e50500-01d3-11eb-8044-2e31658644e1.png" alt="img"/></center>


- Pooling이 없을 때, 4음소(O-kay-Goo-gle) 또는 3음소(Hey-Goo-gle) 패턴을 확인
- Pooling을 사용함으로써 시작부분 보다는 끝부분의 발화가 더 큰 attention weight를 가짐
- LSTM은 이전 상태 값을 누적하여 가지고 있기 때문에 마지막으로 갈수록 더 많은 정보를 가짐으로써 나오게 되는 현상으로 판단


<br/>

---

<br/>

### Ⅴ.  Conclusion 🔎

- 본 논문에서는 keyword 기반의 Text-dependent 화자 검증 시스템을 위한 다양한 Attention mechanism을 실험

- 가장 좋은 방법
1. shared-parameter non-linear scoring function 사용
2. LSTM의 마지막 layer에 divided-layer attention 사용
3. Sliding window maxpooling을 attention weight에 적용

- 위의 3가지를 결합하였을 때 기본 LSTM모델 EER 1.72%에서 14%의 상대적 성능 향상을 가져옴

- <span style="color:#FF0000">**동일한 attention mechanism(특히, shared-parameter scoring function)은 Text-independent한 화자 검증 및 화자 식별을 개선하기 위해 사용될 수 있음**</span>

