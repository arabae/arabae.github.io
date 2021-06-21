---
layout: post
title: "End-to-End DNN based Speaker Recognition Inspired by i-vector and PLDA"
date: 2019-05-22
category: review
thumbnail: /style/image/inspired_ivec.png
use_math: true
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Johan Rohdin, Anna Silnova , Mireia Diez, Oldrich Plchot , Pavel Matejka , Lukas Burget</span>

# 📌 **Abstract**

- 최근 text-dependent뿐만 아니라 짧은 발화에서의 text-independent task에서도 DNN 기반 End-to-End 시스템의 경쟁력을 입증
- 그러나 **긴 발화 text-independent의 경우 아직 i-vector + PLDA 기반 시스템이 더 좋은 성능**을 보임
- <span style="background-color:#fff6dd">i-vector + PLDA baseline을 모방한 speaker verification system을 제안</span>
**(End-to-End 방식으로 훈련되지만 baseline system에 멀리 벗어나지 않도록 정규화)**
- 이러한 방식으로 overfitting으로 발생하는 성능 저하를 해결하였으며, 긴 발화와 짧은 발화에서 모두 i-vector + PLDA baseline system보다 성능이 향상된 것을 확인

<br/>

---

<br/>

# **Ⅰ. Introduction**

[ 이전에 소개된 DNN 기반의 speaker recognition system 특징 ]  
1. **i-vector + PLDA system의 구성요소**(feature extraction, calculation of sufficient statistics, i-vector extraction or PLDA) 중 **하나를 NN**(Neural Network)로 **대체**하거나 **개선**  
   - MFCC feature 대신 bottleneck feature 사용
   - sufficient statistics 계산 시 GMM-UBM 대신 NN acoustic model 사용
   - PLDA를 보완하거나 대체하는 NN 사용

2. **Speaker ID를 분류하여 훈련한 NN**을 통해 **speaker embedding 추출** - 대표적인 특징 : d-vector, x-vector와 같은 embedding  
   - acoustic feature를 입력으로 넣어서 speaker label과 loss를 계산한 뒤, NN 모델의 일부(hidden layer, TDNN + fully-connected DNN 중 DNN)을 utterance-level의 feature로 사용
   - text-dependent, 짧은 발화 text-independent에서 효과적
   - 비교적 긴 발화 text-independent에서는 i-vector + PLDA보다 성능이 낮음

<br/>

<span style="background-color:#dee03f">Proposed Method : i-vector + PLDA baseline을 모방한 End-to-End speaker verification 시스템</span>

**1. f2s** (sufficient statistics 추출을 위한 NN 모듈)

**2. s2i** (i-vector 추출을 위한 NN 모듈)

**3. DPLDA** (점수 계산을 위한 Discriminative PLDA 모듈)

- 세 개의 모듈이 개별적으로 baseline을 모방하고, 훈련되며 이후 결합한 뒤 짧은 발화와 긴 발화 모두에 대해 End-to-End 방식으로 추가 훈련을 진행함

- 이때, 추가 훈련 시 개별적으로 훈련하여 얻은 파라미터가 너무 많이 수정되지 않도록 정규화를 실시 (baseline과 너무 달라지는 것을 방지하고 overfitting의 위험을 줄이는 장점이 존재)

- NIST SRE에서 파생된 3개의 다른 데이터 셋에 대해 시스템을 평가 (다양한 언어의 음성을 포함하고, 2분 미만의 긴 발화와 40초 미만의 짧은 발화 모두에 대해 성능을 테스트)

  <br/>


---


<br/>

# **Ⅱ. Database and Baseline Systems**

1. 훈련 및 테스트는 PRISM  dataset 기반, 3가지 평가 셋
   (1) NIST SRE 2005~2010년 데이터 원본(긴) 전화 발화 중 여성
   (2) (1) 음원을 여러 짧은 발화로 생성(등록 : 20~50초, 테스트 30~40초)
   (3) NIST SRE 2016 평가 세트 (남/여 모두, 단일 등록)

<br/>

2. Generative(PLDA) and Discriminative(DPLDA) Baseline
- 특징 : 60dimension-MFCC (20차원, ∆, ∆∆)
- 훈련 데이터 중 전화 데이터만 사용 (짧은 발화 시간은 10~60초 사이 균일 분포를 따르며 총 85,858개 중 짧은 발화는 22,766개)
- PLDA/DPLDA : 2048개 component를 갖는 UBM, 400차원 i-vector

<br/>

<span style="background-color:#f4d451">PLDA</span>
   - i-vector의 평균(모든 훈련 데이터의 i-vector 평균) 과 길이를 정규화
   - 추가적인 domain 적응이나 score normalization은 수행하지 않음
   - 각 화자가 6개의 발화를 갖도록 훈련 데이터를 68,994개로 줄여서 사용

<span style="background-color:#f4d451">DPLDA</span>
- LBFGS optimizer로 binary cross-entropy를 최적화 (모델 훈련 시, 초기화로 PLDA를 사용)
- i-vector의 평균과 길이를 정규화
- LDA를 수행하여 i-vector의 차원을 250차원으로 축소

<br/>

---

<br/>

# **Ⅲ. Proposed End-to-End DNN Architecture**

**1. Feature to Sufficient statistics : f2s [특징 벡터 → 충분 통계량]**

- 입력 발화의 각 frame에 대해 GMM responsibilities (posteriors, 사후 확률)을 예측
  - 60차원의 MFCC를 전처리(preprocessing) 하여 input으로 사용  
  - 현재 frame을 기준으로 ±15 frame을 고려 (총 31개 frame) → 6개 사용  
  - 6 * 60 → 360차원  

- Hidden layer : 4개 (activation function : sigmoid, node : 1500개)
- Output : 2048개 (GMM-UBM baseline의 component 수) - softmax
- Optimizer : SGD(stochastic gradient descent)
- Loss : categorical cross-entropy (label : GMM-UBM의 사후 확률)
- frame을 충분 통계량으로 pooling (전체 frame에 걸쳐 softmax layer에서 나온 사후 확률, 전처리하지 않은 MFCC)

<br/>

**2. Sufficient statistics to i-vectors : s2i [충분 통계량 → i-vector]**

- f2s에서 나온 충분 통계량을 input으로 사용 (2048x60차원)
- MAP 적응된 supervector로 변환 (112880 차원) - 차원 수를 줄이기 위해 PCA를 사용하여 4000차원으로 축소
- Hidden layer : 3개 (activation function : tanh, 1-2 layer node : 600개, 3 layer node : 250개) - 마지막 layer에서 i-vector 길이 정규화
- NN의 output과 LDA를 사용하여 250차원으로 줄이고 길이를 정규화한 reference i-vector의 average cosine distance
- Optimizer : SGD, L1-regularization

<br/>

<center><img src="https://user-images.githubusercontent.com/46676700/89503150-1e1ceb00-d801-11ea-9660-b2255b35a32f.png" alt="img"/></center>

<br/>

**3. i-vector to scores (DPLDA)**

- 두 i-vector(ϕ 표기)가 주어졌을 때, PLDA 모델의 Log-Likelihood Ratio(LLR)

<center><img src="https://user-images.githubusercontent.com/46676700/89503158-1fe6ae80-d801-11ea-9705-f8eeb4eefac9.png" alt="img" style="zoom:60%;" /></center>  

- DPLDA는 위의 식에서 파라미터 (Λ, Γ, c, k)를 훈련하여 구하는 것
- 두 i-vector가 같은 화자 인지 판단 (Binary cross-entropy 혹은 SVM 최적화를 통해 얻어짐)
- 모든 훈련 데이터를 기반 계산 (전체 batch를 사용) 하나, End-to-End 시스템 훈련 시에는 너무 많은 메모리와 시간이 필요하여 훈련 데이터 중 무작위로 subset을 선택한 Minibatch 기반

<br/>

< Minibatch 선택 rule >
1. 각 화자에 대해 랜덤하게 발화를 쌍으로 만든다
  - 만약 어떤 화자의 발화가 하나라면 동일한 발화를 쌍으로 만들어서 사용
  - 만약 어떤 화자의 발화가 균등한 개수가 아니라면 발화의 쌍 중 하나는 세 개의 발화를 가짐  

2. 각 Minibatch에 대해 임의로 N 개의 발화를 선택하여 선택된 발화로 형성될 수 있는 모든 실험에 사용(마지막 쌍을 선택한 경우 다시 1로 돌아감)

<br/>

**4. End-to-End System**

- 앞서 개별적으로 훈련한 뒤 결합하여 End-to-End로 추가 훈련

- 메모리가 굉장히 많이 필요하는 문제점이 존재

1. PCA : f2s와 s2i를 연결하기 위해 network의 일부가 되어야 하는데, 122800x4000개의 파라미터가 필요  
   - 전체 End-to-End 훈련 전에, s2i NN과 DPLDA 모델만 공동으로 훈련

   - s2i의 개별 훈련 시, f2s가 업데이트되지 않는 이상 입력이 고정이므로 PCA를 거친 특징을 입력으로 사용할 수 있음

<br/>

2. f2s : DPLDA 모듈을 훈련하기 위해 여러 가지 다양한 발화의 모든 frame을 한 번에 처리해야 함  
   - 중간 결과를 메모리에 덜 유지하도록 훈련과정을 수정
   - 하나의 발화에 대해 충분 통계량을 계산하고 block A의 모든 layer의 출력을 없앰
   - block A의 파라미터는 전체 frame(nf) x (1500+1500+1500+1500+2048) 개 변수가 메모리에 저장
   - 충분 통계량 F, N으로 pooling 한 뒤 파라미터 : 2048x60
   - Optimizer : ADAM
   - Training rate를 epoch에서 $C^{prm}_{min}$이 개선되지 않을 때  마다 절반으로 줄임
   - 훈련 데이터는 DPLDA와 같음

---

<br/>

# **Ⅳ.  Results and Discussion**

- Baseline의 일부만 NN으로 대체된 시스템, End-to-End 결과 표

<center><img src="https://user-images.githubusercontent.com/46676700/89503694-df3b6500-d801-11ea-80f2-beb566897aa9.png" alt="img" /></center>

<center>

> 1,2행 : PLDA와 DPLDA baseline의 성능  
> 3행 : UBM이 f2s NN으로 대체되었을 때 성능  
> 4행 : i-vector 추출기가 s2i NN으로 대체되었을 때 성능  
> 5행 : UBM의 충분 통계량 대신 f2s 모듈의 출력으로 s2i 훈련 한 성능  
> 6행 : PLDA 대신 DPLDA를 사용하였을 때 성능  
> 7행 : s2i와 DPLDA만 공동으로 훈련될 때의 성능  
> 8행 : 모든 모듈이 공동으로 훈련될 때의 성능  

</center>

<br/>

- 3개의 모듈이 공동으로 훈련될 때의 성능(8행)과 2개의 모듈이 공동으로 훈련되었을 때 성능(7행)이 큰 차이가 없었음

<br/>

<span style="background-color:#ceddf2">< 3가지 가능성 ></span>
1. Minibatch가 안정적인 훈련을 하기에 너무 작을 수 있다. (3개의 모듈을 공동으로 훈련 시, N=75 최대)  
2. 모델이 local minimum으로 고정될 수 있다. (f2s의 출력에 따라 후속 모델들도 훈련이 되기 때문)  
3. f2s의 설계가 상당히 제약적이다. (사후 확률만 추정 할 뿐 통계 계산에 사용되는 특징을 수정할 수 없기 때문)

<br/>

# **Ⅴ.  Conclusion**

- 다양한 언어와 긴 발화, 짧은 발화를 모두 포함하는 세 개의 서로 다른 데이터셋에 대한 i-vector + PLDA baseline을 능가하는 End-to-End 화자 검증 시스템 개발
- i-vector + PLDA 시스템과 비슷하게 동작하도록 제한함으로써 End-to-End 시스템의 성능을 저하시키는 overfitting을 완화
- 시스템 3개의 서브 모듈 중 3개의 모듈의 공동 훈련은 성능이 좋았지만, 모두 공동 훈련하였을 때 효과적이지 않았음
  - 세가지 모듈을 공동으로 훈련하였을 때 더 나은 성능이 나오도록 개발할 것
- 단일 등록을 사용하도록 설계, 여러 등록을 처리하도록 확장할 것
