---
layout: post
title: "Generalized End to End Loss For Speaker Verification"
date: 2019-07-10
category: review
thumbnail: /style/image/GE2E.png
use_math: true
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno</span>

# 📌 **Abstract**

- 새로운 loss function(Generalized End-to-End, GE2E) 제안
- 본 저자들이 이전 논문에서 제안하였던 Tuple-based End-to-End (TE2E) loss function보다 speaker verification 모델을 더 효율적으로 훈련
- EER을 10%이상 감소 시키면서, 동시에 훈련 시간을 60%까지 단축
- 또한 다중 키워드 ("OK google", "Hey google")와 여러 방언을 지원하는 Domain 적응을 위한 MultiReader 기술을 소개
<br/>

---

<br/>

# **Ⅰ. Introduction**

### **1.1  Background**

**✔ Speaker Verfication**

- 화자의 알려진 발화 (등록 발화, Enrollment)를 기반으로 테스트 발화가 특정 화자에 속하는지 확인

<br/>

**✔ TD-SV / TI-SV**
- 등록과 검증에 사용되는 발화의 제한에 따라 두 가지 카테고리로 나뉨
- TD-SV : Text-Dependent Speaker Verification (같은 내용을 발화)
- TI-SV : Text-Independent Speaker Verification (다른 내용을 발화)

<br/>

**✔ i-vector system**

- TD-SV와 TI-SV에서 모두 효과적인 접근 방식으로, 최근 Nerual Network를 이용하여 이를 대체하는데 집중
- Nerual Network를 이용해 추출한 vector를 embedding vector(d-vector) 라고 하며 i-vector와 유사하게 고정 차원으로 표현이 가능

<br/>

### **1.2 TE2E**

**✔  LSTM network**

<center><img src="https://user-images.githubusercontent.com/46676700/94104922-2a254080-fe73-11ea-9685-5a2244dce1c9.png" alt="img" /></center>

- 각 훈련 단계에서, 하나의 테스트용 발화 𝒙𝑗~와 등록 발화 𝒙𝑘𝑚 tuple을 입력으로 사용

> $x$: 고정 길이의 log melfiterbank  
> $j, k$: 발화한 화자  
> ($j$ 와 $k$ 는 같을 수 있음만약 $x_{𝑗\tilde{}}$와 𝑀개의 등록 발화가 같은 화자라면 tuple positive ($𝑗=𝑘$), 다르면 negative  

<br/>

- 각 tuple에 대해 LSTM output을 L2 정규화
> ${𝒆_𝑗~,(𝒆{k_1},…,𝒆_{k_M})}$  - $e$ : *embedding*

<br/>

- Tuple의 centroid는 M개의 발화로부터 생성한 voiceprint

<center><img src="https://user-images.githubusercontent.com/46676700/94106000-770a1680-fe75-11ea-9da3-ec0fb39d3869.png" alt="img" style="zoom:60%;" /></center>

<br/>

- Cosine Simliarity Function

<center><img src="https://user-images.githubusercontent.com/46676700/94106049-91dc8b00-fe75-11ea-980c-5e7b6d001a22.png" alt="img" /></center>
> 𝑤,𝑏 는 학습되는 변수

<br/>

- TE2E loss

<center><img src="https://user-images.githubusercontent.com/46676700/94106059-93a64e80-fe75-11ea-8b31-a8b2629cfc6f.png" alt="img" style="zoom:80%;" /></center>
<center><img src="https://user-images.githubusercontent.com/46676700/94106117-ae78c300-fe75-11ea-8bfb-cff4fa7b2e8a.png" alt="img" style="zoom:80%;" /></center>

> $\sigma(𝑥) = 1/(1+𝑒^{−𝑥})$ : sigmoid function   
> $\delta(j, k) = 1~~~(𝑗=𝑘)~~or~~0~~~(𝑗≠𝑘)$

<br/>


---


<br/>

# **Ⅱ. GE2E Model**

### **2.1 Training Method**

**✔  GE2E training**

- Fig. 1에 나타난 것과 같이 N( 화자 수 ) X 발화 수 batch 형태로 많은 수의 발화를 한번에 처리

<center><img src="https://user-images.githubusercontent.com/46676700/94107053-a588f100-fe77-11ea-9812-931dfe797405.png" alt="img" style="zoom: 80%;" /></center>

> $x_{ji}$: 화자 $j$ 의 $i$ 번째 발화를 추출한 특징 벡터  
> $𝑓(𝒙_{ji}; 𝒘)$: LSTM 과 linear layer 를 거치고 나온 마지막 출력  
> $e_{ji}$: L2 정규화 후 embedding 벡터

<center><img src="https://user-images.githubusercontent.com/46676700/94107148-d2d59f00-fe77-11ea-80f8-973a38b624f0.png" alt="img" style="zoom: 80%;" /></center>

<br/>

<center><img src="https://user-images.githubusercontent.com/46676700/94107170-dff28e00-fe77-11ea-87a5-be7ed870bfc8.png" alt="img" style="zoom:40%;" /></center>

> Embedding vector $e_{ji}$, 모든 centroid $c_k$로계산 $(1≤𝑗,𝑘≤𝑁,1≤𝑖≤𝑀)$  
> $𝑤 > 0$ : cosine similarity 값이 클수록 similarity 를 크게 하기 위하여 양수로 설정

<br/>

**✔  TE2E와 GE2E의 차이점**

- TE2E의 similarity (equation 2 는 embedding vector 𝒆𝑗~와 하나의 tuple centroid 𝒄𝑘사이의 유사함을 계산 (scalar)
- GE2E (equation 5)는 각 embedding vector 𝒆𝑗𝑖와 모든 중심 𝒄𝑘의 유사함을 계산하여 행렬로 구축

<br/>

**✔  목적**

- 훈련동안 , 각 발화의 embedding 이 본인 발화의 centroid 와는 유사함과 동시에 다른 화자의 centroid 와의 거리는 멀게 (fig 1. 에서 색상의 값은 크고 회색의 값은 작게
- Fig. 2에서 파란색 embedding vector 가 그 화자의 centroid(파란색 삼각형)과 거리는 가까우며 , 다른 화자의 centroid(빨간색, 보라색 삼각형) 과는 거리가 멀도록

<center><img src="https://user-images.githubusercontent.com/46676700/94107458-7921a480-fe78-11ea-886e-a847186d8e93.png" alt="img" style="zoom:60%;" /></center>

<br/>

**✔  Softmax - Similarity matrix**

- $S_{ji,k}$에 softmax 를 적용하여 j 와 k 가 같은 화자 일 경우는 출력 값을 1, 다른 화자일 경우 0 이 되도록 함
- 각 embedding vector 를 그 화자의 centroid 와는 가깝게 하고 , 다른 화자의 centroid 로
부터는 멀어지게 함

<center><img src="https://user-images.githubusercontent.com/46676700/94107622-d4539700-fe78-11ea-8d58-18bea7a203e9.png" alt="img" style="zoom:80%;" /></center>

<br/>

**✔  Contrast - Similarity matrix**

- Contrast loss는 positive 쌍과 가장 negative 한 쌍으로 정의
- 모든 발화에 대해 두 가지 구성요소가 loss 에 추가

  (1) embedding vector와 그 화자의 voiceprint 사이의 positive 일치  
  (2) 다른 화자들의 발화 중 가장 높은 유사성을 갖는 voiceprint 의 negative 일치

<center><img src="https://user-images.githubusercontent.com/46676700/94107626-d61d5a80-fe78-11ea-8640-b4e3f492149f.png" alt="img" style="zoom:80%;" /></center>

<br/>

**✔  Softmax&Contrast - Similarity matrix**

- TI-SV 의 경우 softmax loss 가 약간 더 나은 성능을,, TD SV 의 경우 contrast loss 가 더 나은 성능을 보여 두 가지 GE2E loss 의 구현이 모두 유용함을 발견
- $e_ji$제거 : 화자의 centroid 계산시 , 훈련이 안정되고 사소한 문제를 피할 수 있도록 도와줌
- j 와 k 가 같은 화자일 경우는 (1) 대신 (8) 을 사용하여 centroid 계산

<center><img src="https://user-images.githubusercontent.com/46676700/94108065-95721100-fe79-11ea-90cf-b8fe4dd86f1c.png" alt="img" style="zoom:80%;" /></center>

<br/>

**✔  Eq. 4, 6, 7, 9를 합하여 만든 최종 GE2E loss**

<center><img src="https://user-images.githubusercontent.com/46676700/94108143-bc304780-fe79-11ea-9c40-d584284261bd.png" alt="img" style="zoom:80%;" /></center>


<br/>


---

<br/>

### **2.2 Comparison between TE2E and GE2E**

**✔  모든 입력 𝒙𝑗𝑖에 대해 TE2E loss 에서 발생하는 tuple 의 수**

- (1) positive tuples : 화자 j 에서 무작위로 P 발화 선택

<center><img src="https://user-images.githubusercontent.com/46676700/94108310-09acb480-fe7a-11ea-9b77-bbc90e239545.png" alt="img" style="zoom:70%;" /></center>

- (2) negative tuples : 화자 j 와 다른 화자 k 의 발화에서 무작위로 P 발화 선택

<center><img src="https://user-images.githubusercontent.com/46676700/94108326-1204ef80-fe7a-11ea-9252-2f0510d28f49.png" alt="img" style="zoom:70%;" /></center>

**⭐ 총 TE2E loss의 tuple 수**

<center><img src="https://user-images.githubusercontent.com/46676700/94108404-3365db80-fe7a-11ea-82b4-2674279c20ba.png" alt="img" /></center>


**✔  GE2E loss 의 tuple 수**

- 각 화자의 모든 발화를 선택하여 centroid 로 계산하므로 P=M
- 따라서 TE2E 의 최소 값 [2x(N-1)]과 동일
- <span style="background-color:#f2cfa5">**GE2E가 TE2E 보다 짧은 시간 내에 더 나은 모델로 수렴**</span>

<br/>

### **2.3 Training with MultiReader**

**✔  작은 데이터 셋 D1과 큰 데이터 셋 D2 존재**

- D1 domain model 에 관심이 있는데 , 동일한 domain 은 아니지만 더 큰 D2 dataset 이 있을 때 , **D2 의 도움을 받아** dataset D1 에서 우수한 성능을 보이는 단일 모델을 교육하고자 함

<center><img src="https://user-images.githubusercontent.com/46676700/94109181-9310b680-fe7b-11ea-82df-c7db9d057167.png" alt="img" /></center>

- Regularization 기법과 유사

<center><img src="https://user-images.githubusercontent.com/46676700/94109049-60ff5480-fe7b-11ea-9be6-929668d7d5b1.png" alt="img" style="zoom:60%;"/></center>

- D1에 충분한 데이터가 없을 때 , D2 에서도 잘 수행할 수 있도록 함으로써 overfitting 이 발생하는 것을 방지

**✔  일반화**

- K 개의 다른 data source : 𝐷1,…,𝐷𝐾를 결합하기 위해 일반화
- 각 data source 의 가중치 𝛼𝑘를 할당하여 해당 data source 의 중요성 표시

<center><img src="https://user-images.githubusercontent.com/46676700/94109317-c2272800-fe7b-11ea-88d3-dff6b42d6d0a.png" alt="img" style="zoom:90%;"/></center>

---

<br/>

# **Ⅲ. Experiments**

**✔  참조논문 [6]과 process 동일**

- 100개 frame을 30 frame씩 (10 frame 옆으로 이동해가며 ) 사용
- 40-dimensional log mel filterbank
- LSTM 3-layer + projection (d-vector size)

<br/>

**✔  Hyper Parameter**

- N : 64 (speakers), M : 10 (utterances)
- Learning rate : 0.01 (30M 단계마다 절반씩 감소
- Optimizer : SGD
- Loss function의 좋은 초기 값 : (w, b) = (10, 5)

<br/>

### **3.1 TD-SV**

- Keyword detection 과 speaker verification 같은 특징 사용
- Keyword detection 은 keyword 가 포함된 frame 만 SV system 으로 전달
- 이 frame 은 고정 길이 segment 형성
- Hidden node : 128, Projection size : 64

<br/>

**✔ Multiple Keyword**

- 사용자들이 동시에 여러 개의 키워드 지원을 더 선호하여, "Ok google"과 "Hey google" 지원
- 하나의 구절로 제한되거나 완적히 제약되지는 않기 때문에 여러 keyword로 speaker verification하는 것은 TD-SV와 TI-SV 사이에 놓임
- 여러 data source를 직접 혼합하는 것과 같은 단순한 접근 방식에 비해 MultiReader는 data source의 크기가 불균형한 경우에 사용할 수 있는 등 큰 이점을 갖음

> ~150M 발화, ~630K 화자로 이루어진 "Ok google" set과 ~1.2M 발화와 ~18K로 이루어진 "Hey google" set을 비교하면 "Ok google"이 125배 발화 수가 더 많으며 35배 화자 수가 더 많음

<br/>

**✔ 평가 방법 및 결과**

- 4가지 경우에 대해 EER 측정 (2개의 keyword로 나올 수 있는 조합)
- 테스트 dataset : 665 명의 화자 / 평균 4.5 회 등록 발화 , 10 개의 테스트 발화
- MultiReader를 적용한 것이 4 가지 경우 모두에서 약 30% 의 상대적 성능 향상을 보임

<center><img src="https://user-images.githubusercontent.com/46676700/94112150-00264b00-fe80-11ea-9b21-fc9d738e90de.png" alt="img"/></center>

<br/>

**✔ 추가 실험**

- ~83K 서로 다른 화자와 환경 조건의 대규모 dataset (평균 7.3회 등록, 5개 테스트 발화 사용)
- GE2E model은 TE2E보다 약 60% 더 적은 훈련 시간을 소모

<center><img src="https://user-images.githubusercontent.com/46676700/94112311-349a0700-fe80-11ea-8edf-f098ab62b9a8.png" alt="img" style="zoom:75%;"/></center>

> 1 행 : 512 개의 hidden node 와 128 차원의 embedding vector 크기를 가진 단일 계층 LSTM  
> 2행  : 3 layer LSTM (TE2E)  
> 3 행 : 3 layer LSTM (GE2E)

<br/>

### **3.2 TI-SV**

- Hidden node : 768, Projection size : 256
- VAD(Voice Activity Detection) 후 고정 길이 segment로 나눔 ; partial utterances

<br/>

**✔ Train**

- 각 데이터 batch 에 대해 𝑙𝑏,𝑢𝑏] = [140, 180] frame 내에 임의로 시간 길이 t 선택
- 해당 batch 내 모든 발화의 길이는 t가 되어 고정 길이의 segment를 갖음

<center><img src="https://user-images.githubusercontent.com/46676700/94110427-97d66a00-fe7d-11ea-94f6-7c3f2413e535.png" alt="img" style="zoom:80%;"/></center>

<br/>

**✔ Test**

- window size만큼 고정 segment를 가져와 d-vector 추출
- window size를 50%만큼 겹치게 sliding하여 이동
- window마다 추출된 d-vector를 L2 정규화하고 average하여 최종 d-vector를 생성

<center><img src="https://user-images.githubusercontent.com/46676700/94110496-b2104800-fe7d-11ea-96b1-41d04121b01d.png" alt="img" style="zoom:60%;"/></center>

<br/>

**✔ 실험 결과**

- 훈련 dataset : 36M 발화와 18K 화자를 사용
- 테스트 dataset : 1000 명의 화자 , 평균 6.3 개 등록 발화 , 평균 7.2 개의 테스트 발화 사용

<center><img src="https://user-images.githubusercontent.com/46676700/94113342-b8082800-fe81-11ea-9cde-7831829befa1.png" alt="img" style="zoom: 50%;" /></center>

> Softmax : 훈련 데이터의 모든 화자에 대한 label 을 예측  
> TE2E : TE2E 로 훈련된 모델  
> GE2E : GE2E 로 훈련된 모델

<br/>

---

<br/>

# **Ⅳ.  Conclusion**

- Speaker Verification 을 위한 보다 효율적인 GE2E loss function 제안
- 이론 및 실험적 결과 에서 모두 본 논문에서 제안한 모델의 장점 을 입증
- 다양한data source 를 결합하는 MultiReader 기법을 도입하여 여러 키워드와 언어를 지원 할 수 있도록 함
- <span style="color:#FF0000">**두 가지 기법을 결합하여 보다 정확한 Speaker Verification Model 구축**</span>
