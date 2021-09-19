---
layout: post
title: "ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context"
date: 2021-08-04
category: review
thumbnail: /style/image/contextnet.png
use_math: true
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Wei Han, Zhengdong Zhang, Yu Zhang, Jiahui Yu, Chung-Cheng Chiu, James Qin, Anmol Gulati, Ruoming Pang, Yonghui Wu</span>

# **<span style="color:#E84560">_Abstract_</span>**
End-to-End speech recognition에서 Convolution neural network (CNN)는 기대되는 결과를 보여주지만, 여전히 RNN/Transformer 기반 모델보다는 뒤쳐짐

본 논문에서는 지금까지 없었던 ContextNet이라 부르는 CNN-RNN-transducer architecture로 이러한 차이를 메우고, 더 나아갈 방법을 연구

ContextNet는 **squeeze-and-excitation module을 추가**하여 global context information을 통합하는 **fully convolutional encoder**가 특징

추가적으로 computation과 accuracy 사이의 좋은 trade-off를 달성하는 ContextNet의 width scale을 간단하게 scaling 방법을 제안

전반적으로 Librispeech benchmark를 사용하여 실험

**_ContextNet_**

- LM X: WER 2.1%/4.6% (clean/nosiy test set)
- LM O: WER 1.9%/4.1%
- 10M parameter: 2.9%/7.0%

**_비교 모델_**

- LM O: WER 2.0%/4.6%
- 20M parameter: 3.9%/11.3%

⇒ 제안된 ContextNet 모델의 우수성은 훨씬 더 큰 내부 데이터 세트에서도 확인됨


# **<span style="color:#E84560">_1. Introduction_</span>**

E2E speech recognition를 위한 CNN 기반 모델은 관심이 점점 증가하는 추세

그 중에서도 **Jasper model**은 LibriSpeech test-clean에서 WER 2.95%로 SOTA를 달성

> **_Jasper model_**

- 1D convolution layer를 쌓은 deep convolution기반 encoder
- skip conntection

> **_Depthwise separable convolutions (Xception)_**

- CNN 모델의 속도와 정확도를 더욱 높이기 위해 사용

CNN기반 모델의 핵심 이점은 parameter를 효율적으로 사용

→ 그러나 CNN best 모델(QuartzNet)은 여전히 RNN/transformer기반 모델보다 뒤쳐짐

*RNN/transformer 기반 모델과 CNN 모델의 **주요 차이점**은* **context의 길이가 다름**

- Bidirectional RNN model에서 이론상 cell은 전체 시퀀스의 정보에 접근 가능
- Transformer 모델에서 attention mechanism은 먼 time stamp에 있는 두 개의 노드가 서로 영향을 주도록 명시적으로 허용
- 제한된 kernel size를 갖는 naive한 convolution은 time domain에서 작은 window만 cover할 수 있음

→ context가 작고, global information을 통합할 수 없음

본 논문에서 CNN기반 ASR 모델과 RNN/transformer 기반 모델의 WER 차이의 핵심적인 이유인 **global context**의 부족을 연구

→ CNN 모델의 global context를 강화하기 위해 [12]에서 소개된 **sequeeze-and-excitation (SE) layer**에서 영감을 얻어, **ContextNet**이라고 부르는 ASR을 위한 이전에 없던 CNN 모델을 제안

> **_Sequeeze-and-Excitation (SE) layer_**

- local feature vector sequence를 single global context로 압축하고, 이 context를 다시 각 local feature vector로 broadcast 후 곱셈을 통해 둘을 병합
    - average 등을 통해 하나의 global feature를 생성하고, 다시 차원을 local feature로 broadcast한 뒤 둘을 곱함으로써 하나의 결과를 얻음

naive한 convolution layer 뒤에 SE layer를 배치할 때 convolution output에 global 정보에 대한 접근을 줌

경험적으로 **ContextNet에 SE layer를 추가**하면 LibriSpeech test-other에서 **WER이 가장 많이 감소**한다는 것을 발견

# **<span style="color:#E84560">_2. Model_</span>**

### **<span style="color:#724598">2.1 End-to-End Network: CNN-RNN-Transducer</span>**

— ***ContextNet high-level design***

- RNN-Transducer framework 기반
- network의 3가지 구성 요소
  1. input utterance를 받는 audio encoder
  2. input label을 받는 label encoder
  3. 이 둘을 결합하고 decoding하는 joint network

⇒ LSTM 기반 label encoder와 joint network를 사용하지만 **새로운 CNN 기반 audio encoder를 제안**

### **<span style="color:#724598">2.2 Encoder Design</span>**

— ***Convolutional encoder (model accuracy maintaining, temporal length reduce)***

$x = (x_1,...,x_T)$ ; input sequence  
$h = AudioEncoder(x) = C_K(C_{K-1}(...C_1(x)))$  

- encoder는 원래 signal x를 high level representation으로 변환
- 각 $C_k(·)$는 Convolution block을 정의
    - 몇 개의 convolution layer가 포함되어 있으며, 각 layer에는 batch normalization, activation function이 있음
    - 또한, Squeeze-and-excitation 구성 요소와 skip connection을 포함

C(·)의 중요한 모듈
![image](https://user-images.githubusercontent.com/46676700/133924656-7617477a-2f82-4f32-a1c9-754587dcfc33.png)


> **2.2.1. seqeeze-and-excitation**

$SE(·)$ ; Squeeze-and-Excitation function

- input x에 대해 global average pooling → global channelwise weight $\theta(x)$ → element-wise multiplies each frame

⇒ 1D case에 이 idea를 도입

$\bar{x} = \frac{1}{T} \Sigma_t{x_t}$

$\theta{(x)} = Sigmoid(W_2(Act(W_1\bar{x}+b_1))+b_2)$

$SE(x) = \theta{(x)} \; ◦ \; x$

- ◦ ; element-wise multiplication
- $W_1, W_2$ ; weigth matrix
- $b_1, b_2$ ; bias vector

> **2.2.2. Depthwise separable convolution**

$conv(·)$ ; encoder에서 사용되는 convolution function

- depth별 분리 가능한 convolution사용
- 이러한 설계가 **정확도에 영향을 미치지 않고**, **더 나은 parameter 효율성을 달성**하기 위해 다양한 연구[6, 4, 28]에서 보여줌
- **단순화**를 위해 network의 모든 깊이별 convolution layer에서 동일한 kernel size를 사용

> **2.2.3. Swish activation function**

$Act(·)$ ; encoder의 activation function

- ReLU와 swish function을 모두 실험

**Swish function**
$Act(x) = x \cdot \sigma{(\beta x)} = \frac{x}{1+\exp(-\beta x)}$

- $\beta$ ; 모든 실험에 대해 1
- **swish function**이 ReLU보다 일관되게 작동하는 것을 관찰

> **2.2.4. Convolution block**

figure 3 ; $C(\cdot)$의 high-level architecture

- block $C(\cdot)$에는 여러 $conv(\cdot)$이 포함될 수 있음 (m ;  $conv(\cdot)$의 수)
- $BN(·)$ ; batch normalization

→ $f(x) = Act(BN(Conv(x))$

⇒ $C(x) = Act(SE(f^m(x)) + P(x))$

- $f^m$ ; input에 $f(\cdot)$ function이 m개 층이 쌓임
- $P(\cdot)$ ; residual에 대한 pointwise projection function

![image](https://user-images.githubusercontent.com/46676700/133924665-8d0cc2af-3731-4796-a012-567ad1339027.png)

첫 번째, 마지막 layer는 나머지(m-2) layer와 다를 수 있음

- block에 input channel 수($D_{in}$)과 output channel 수($D_{out}$)이 있는 경우, 첫 번째 layer ($f'$)는 $D_{in}$ channel을 $D_{out}$ channel로 변환하고 나머지(m-1) layer는 channel 수를 $D_{out}$으로 유지
- block이 input sequence를 two time에 걸쳐 downsampling하는 경우, 마지막 layer는 stride가 2이며 나머지(m-1) layer는 stride가 1 (그렇지 않으면 모두 stride가 1)

기존 연구들에 따라 projection function $P$는 첫 번째 layer와 같은 stride를 갖음

> **2.2.5. Progressive downsampling**

temporal downsampling을 위해 strided convolution 사용

- downsampling layer가 많을 수록 계산 비용이 감소
- encoder서 과도한 downsampling은 decoder에 부정적인 영향을 미칠 수 있음

→ 경험적으로 progressive하게 8배를 downsampling하는게 속도와 성능면에서 좋은 trade-off를 갖는 것을 발견 (이러한 절충은 섹션 3.3에서 논의)

> **2.2.6. Configuration details of ContextNet**

ContextNet에는 23개의 convolution block $C_0, . . . , C_{22}$ 이 존재

- 모든 convolution block에는 각각 하나의 convolution layer만 있는 $C_0$ 및 $C_{22}$를 제외하고, 5개의 convolution layer 존재

Table 1 ; architecture detail이 요약됨

- global parameter $\alpha$는 model scaling을 조정
    - $α > 1$: $α$를 증가시키면 convolution의 channel 수가 증가 → model size가 클수록 model은 더 많은 표현력을 갖음

![image](https://user-images.githubusercontent.com/46676700/133924661-dffde0a7-9213-46d6-8ad9-d3f76bfab21a.png)

# **<span style="color:#E84560">_3. Experiments_</span>**

- Librispeech dataset
    - 970시간 labeled speech + text only corpus (for building language model)
- feature
    - 10ms stride, 25ms window, 80 dimensional fiterbank
- Adam optimizer
- Transformer learning rate schedule
    - 15k warm-up step, 0.0025 peak learning rate
- L2 regularization with $10^{-6}$ weight
- single layer LSTM as decoder
    - 640 input dimension, variational noise
- SpecAugment
    - mask parameter (F=27), 10개 maximum time-mask ratio (ps=0.05), time-mask의 최대 크기는 utterance길이 * ps
    - time warping 사용 X

LibriSpeech 960h에서 구축된 1k WPM으로 토큰화된 LibriSpeech960h transcript가 추가된 LibriSpeech에서 훈련된 width 4096의 3-layer LSTM LM 사용

- LM은 dev-set transcript에서 word-level perplexity가 63.9
- shallow fusion에 대한 LM weight $λ$는 grid search를 통해 dev-set에서 조정
- 모든 모델은 Lingvo toolkit으로 구현

### **<span style="color:#724598">3.1. Results on LibriSpeech</span>**

LibriSpeech에서 ContextNet의 세 가지 다른 구성을 평가

- 모두 table 1을 기반으로 하지만 network width $α$가 다름 (즉, model size가 다름)

small, medium, large ContextNet에 대해 {0.5, 1, 2}로 α를 선택

- 또한 참조용으로 자체 LSTM baseline을 구축

Table 2는 이전에 발표된 몇 가지 시스템과의 evaluation result 비교를 요약

- 이전에 발표된 시스템에 비해 ContextNet이 개선되었음을 시사함
- ContextNet(S): 언어 모델이 있거나 없는 유사한 크기[4]의 이전 시스템에 대해 개선됨
- ContextNet(M): 31M parameter만 가지고 있으며, 훨씬 더 큰 시스템과 비교하여 유사한 WER을 달성
- ContextNet(L): 이전 SOTA보다 test-clean에서 상대적으로 13%, test-other에서 상대적으로 18% 더 우수함

![image](https://user-images.githubusercontent.com/46676700/133924670-44ad7506-3262-45e1-8601-859c6e230dca.png)

### **<span style="color:#724598">3.2. Effect of Context Size</span>**

> **ablation 연구를 수행**

- ASR용 CNN 모델에 global context를 추가하는 효과를 검증
- Squeeze-and-Excitation module이 LibriSpeech test-clean/test-other의 WER에 어떻게 영향을 미치는지

Table 1의 ContextNet은 모든 squeeze-and-excitation 모듈이 제거되고, $α$ = 1.25가 zero context의 baseline으로 사용

- vanilla seuqeeze-and-excitation module은 전체 utterance를 context로 사용
- 다양한 context size의 영향을 조사하기 위해 squeeze-and-excitation module의 global average pooling 연산자를 pooling window의 크기로 context를 제어할 수 있는 stride-one pooling 연산자로 대체
- 모든 convolution block에서 256, 512 및 1024의 window size를 비교

Table 3; SE module은 baseline에 비해 개선됨

- 또한, context window의 길이가 길어질수록 더욱 향상됨
- 이는 image classification model에 대한 SE의 유사한 연구에서 관찰된 것과 일치 [34].

![image](https://user-images.githubusercontent.com/46676700/133924678-2e019104-c9f1-4296-84a1-c4c53eb6bb8f.png)

### **<span style="color:#724598">3.3. Depth, Width, Kernel Size and Downsampling</span>**

_Depth_: convolutional block의 수에 대해 sweeping을 수행하고, 최상의 config은 Table 1에 있음

- 이 config를 사용해 안정적인 수렴으로 하루만에 모델을 훈련할 수 있음을 발견

_Width_: 모든 encoder layer에서 network width(즉, channel 수)를 전체적으로 확장하고 모델 성능에 미치는 영향을 연구

- 구체적으로, Tabel 1에서 ContextNet 모델을 취하고, $α$를 sweep하여 LibriSpeech에서 model size와 WER 확인 (Tabel 5 ; 결과 요약)
- ContextNet의 WER과 모델 크기 사이의 적절한 균형을 보임

_Downsampling and kernel size_: Tabel 4 ; downsampling 및 filter size의 다양한 선택과 함께 LibriSpeech의 FLOPS 및 WER을 요약

- baseline: $C_3$에 하나의 downsampling layer 추가

    → 따라서 baseline은 2배 temporal reduction

- {3, 5, 11, 21}에서 kernel size를 sweep하고, 각 kernel size는 모든 depth별 convolution layer에 적용

    → progressive한 downsampling이 FLOPS 수를 상당히 절약할 수 있음을 시사

- 또한, 실제로 모델의 정확도에 약간의 이점이 있었으며, progressive downsampling으로 kernel를 늘리면 모델의 WER이 감소함!

![image](https://user-images.githubusercontent.com/46676700/133924687-15f0c85f-a9b2-4629-ac63-a0ef7b849ade.png)
![image](https://user-images.githubusercontent.com/46676700/133924693-c9268573-4396-411b-8f2b-6728af6196ca.png)

**💡FLOPS** (FLoating point Operations Per Second)
컴퓨터 성능을 수치로 나타날 때 주로 사용되는 단위
초당 부동소수점 연산이라는 의미로 컴퓨터가 1초동안 수행할 수 있는 부동소수점 연산의 횟수

**💡GLOPS**
1초의 audio 처리에 대한 average encoder FLOPS

### **<span style="color:#724598">3.4. Large Scale Experiments</span>**

제안된 architecture가 대규모 dataset에서도 효과적

[35]와 유사한 실험 설정을 사용

- training set: [36]의 접근 방식에 의해 생성된 semi-supervised transcript가 포함된 public Youtube video
- 총 24.12시간 동안 117개의 video를 평가 (testset은 다양하고 challenging한 음향 환경)

Table 6 ; 결과 요약

→ ContextNet이 더 적은 parameter와 FLOPS로 convolution과 bidirectional LSTM의 조합인 TDNN(이전 최고의 architecture)를 상대적으로 12% 능가

![image](https://user-images.githubusercontent.com/46676700/133924701-f95e23a4-a2a2-4d5f-8081-160ceb12fd18.png)

# **<span style="color:#E84560">_4. Conclusion_</span>**

- End-to-End speech recognition을 위한 CNN 기반 architecture를 제안하고 평가
- 이전에 발표된 CNN 모델에 비해 훨씬 적은 수의 parameter로 LibriSpeech benchmark에서 더 나은 정확도를 달성
- 제안된 architecture는 network의 width를 제한하여 작은 ASR 모델을 검색하는데 쉽게 사용 가능함
- 훨씬 더 크고, challenge한 dataset에 대한 초기 연구에서도 본 논문 저자들의 발견이 확인됨
