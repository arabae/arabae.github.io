---
layout: post
title: "Conformer: Convolution-augmented Transformer for Speech Recognition"
date: 2021-08-03
category: review
thumbnail: /style/image/conformer.png
use_math: true
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang</span>

# **_Abstract_**

최근 Transformer 및 Convolution neural network(CNN) 기반 모델은 Automatic Speech Recognition(ASR)에서 Recurrent neural networks (RNNs)보다 성능이 좋아 기대되는 결과를 보임

Transformer 모델은 content-based global interaction을 잘 포착하는 반면 CNN은 local feature를 효과적으로 활용함
- parameter-efficient 방식으로 audio sequence의 local 및 global dependency를 모두 모델링하기 위해 CNN과 Transformer를 결합하는 방법을 연구하여 두 세계의 장점을 모두 달성

**⇒ Conformer라는 음성 인식을 위한 Convolution-Augmented Transformer를 제안**

Conformer는 SOTA 정확도를 달성하는 이전 Transformer 및 CNN 기반 모델보다 훨씬 뛰어난 성능을 가져옴

LibriSpeech 벤치마크 사용
- WER 2.1% / 4.3% (language model X) - test/testother
- WER 1.9% / 3.9% (language model O)
- WER 2.7% / 6.3% (small model, only 10M parameter)

# **_1. Introduction_**

NN기반의 End-to-End ASR system은 최근 몇 년 동안 크게 개선됨

RNN은 audio sequence의 temproal dependency를 효과적으로 모델링할 수 있기 때문에 ASR에 대해 사실상 일반적인 선택

최근 self-attention에 기반의 **transformer 구조**는 **long distance interaction을 capture**하는 능력과 **high training efficiency**로 sequence 모델링에 주로 사용됨

더불어, CNN도 **local receptive field layer**를 통해 **점진적으로 local context를 capture**하여 ASR에서도 성공적

그러나 self-attention 또는 CNN 모델은 각각 한계점이 존재

> ***Transformers***

- long-range global context pattern에 효과적
- 세분화된 local feature pattern을 추출하는 능력은 떨어짐

> ***CNN***

- local 정보를 활용하고, vision에서 사실상 computational block으로 사용됨
- [translation equivariance](#further-reading)를 유지하고 edge와 shape과 같은 feature를 capture할 수 있는 local window를 통해 shared position-based kernel을 학습
- local connectivity를 사용하는 것은 global information을 capture하기 위해선 더 많은 layer와 parameter가 필요하다는 제한이 존재

이러한 문제점을 해결하기 위해 동시에 연구된 **contextnet**은 더 긴 context를 capture 하기 위해 **각 residual block에 squeeze-and-excitation module을 둚**
- 그러나 전체 sequence에 대해 **global average만 적용**하기 때문에 **dynamic한 global context**를 capture하기엔 여전히 **제한적**임

최근 연구에 따르면 CNN과 self-attention을 결합하면 개별적으로 사용하는 것보다 향상되었음

- position-wise local feature를 모두 학습하고 content-based global interaction을 사용할 수 있음
- 동시에 [15, 16]과 같은 논문은 equivariance을 유지하는 상대적 위치 기반 정보로 self-attention을 강화함
- Wu et al. [17]은 입력을 self-attention과 convolution의 두 가지 branch로 분할하고 출력을 연결하는 multi-branch architecture를 제안
    - 이 task는 mobile application을 대상으로 했으며, machine translation task의 개선을 보여줌

<center><img src="https://user-images.githubusercontent.com/46676700/128826541-f87104f7-5b5e-41c9-9081-29db15b294bf.png" alt="img" style="zoom:40%;"/></center>

본 논문에서는 ASR에서 CNN과 self-attention을 유기적(organically)으로 결합하는 방법을 연구
global과 local interaction이 parameter 효율성을 위해 중요하다고 가정
→ 이를 달성하기 위해 self-attention과 convolution의 새로운 조합이 두개의 장점을 모두 달성할 것이라고 제안

self-attention은 global interation을 학습하는 반면 convolution은 relative-offset-based local correlation를 효율적으로 capture함
- Wu et al. [17, 18],에서 영감을 받았고, 그림 1과 같이 한 쌍의 feedforward module 사이에 끼워진 self-attention과 convolution의 새로운 조합을 소개!

> ***Conformer***

이전 SOTA Transformer Transducer[7]와 비교
- LibriSpeech dataset 사용 (외부 language model이 있는 testother 데이터 셋에서 상대적으로 15% 향상)

10M, 30M, 118M parameter 크기를 갖는 모델 비교
- 10M: test/testother에서 2.7%/6.3%로 유사한 크기의 다른 모델[10]과 비교했을 때 개선됨
- 30M: 139M parameter를 사용하는 transformer transducer[7]보다 개선됨
- 118M: 언어 모델을 사용하지 않고 2.1%/4.3%, 사용하면 1.9%/3.9% 성능을 보임

➕ attention head 수, convolution kernel size, activation fuction, feedforward layer 배치, convolution module을 transformer기반 network에 추가하는 다양한 방법의 효과에 대해 깊이 연구하고, 각각이 어떻게 정확도를 향상시키는지 초점을 둚

<img src="https://user-images.githubusercontent.com/46676700/128826558-cf9ff480-0a20-4313-804d-569ac4c39e3e.png" alt="img" style="zoom:60%;"/>

# *2. ConformerEncoder*

audio encoder는 먼저 convolution subsampling layer을 사용해 입력을 처리하고, 다음에 fig1과 같이 여러 conformer block을 거침

본 논문 model의 구별되는 특징은 [7, 19]에서 transformer block 부분이 conformer block으로 사용됨

conformer block은 4개의 module(feed-forward module, self-attention module, convolution module, second feed-forward module)이 함께 쌓여 구성됨

section 2.1, 2 and 2.3에서는 각각 self-attention, convolution, feed-forward module을 소개하고, 마지막으로 2.4에서는 이러한 하위 block이 어떻게 결합되는지 설명

### 2.1. Multi-Headed Self-Attention Module

relative sinusoidal(sin 곡선) positional encoding 방식인 Transformer-XL의 중요한 기술을 통합하면서 multi-head self-attention (MHSA)를 사용

**💡 relative positional encoding**  
- self-attention module이 다른 입력 길이에 대해 더욱 잘 일반화할 수 있도록 함
- resulting encoder는 발화 길이의 변화에 대해 더 강인함

더 깊은 모델을 훈련하고, 정규화하는데 도움이 되는 dropout과 함께 pre-norm residual unit을 사용함

아래의 그림 3은 multi-head self-attention module block을 나타냄

<center><img src="https://user-images.githubusercontent.com/46676700/128826564-520cebdc-c97e-45b1-8349-2842c44f6ca0.png" alt="img" style="zoom:40%;"/></center>

### 2.2. Convolution Module

[17]에서 영감을 받아 convolution module은 pointwise convolution과 gated linear unit(glu)인 gating mechanism으로 시작

그 다음 1D depthwise convolution layer가 이어지고, Batchnorm은 deep 모델 훈련을 돕기 위해 convolution 직후에 위치함

그림 2는 convolution block을 나타냄

<img src="https://user-images.githubusercontent.com/46676700/128827668-4697e2e9-3d33-49e7-9968-28a8af2a70e8.png" alt="img" style="zoom:60%;"/>

### 2.3. FeedForward Module

[6]에서 제안된 Transformer 구조는 MHSA layer 이후 feed-forward module이 이어지고, two linear transformation 사이에 nonlinear activation이 존재함

residual connectiondms feed-forward layer 위에 추가되고 layer normalization이 이어짐

이 구조는 Transformer ASR model [7, 24]에도 적용됨

<img src="https://user-images.githubusercontent.com/46676700/128826571-ee6a4944-a20c-4625-98ba-99df6b0fc53c.png" alt="img" style="zoom:60%;"/>

pre-norm residual unit[21, 22]을 따르고, residual unit안에 첫 번째 linear layer 이전 입력에서 layer normalization을 적용함

또한, Swish activation 및 dropout을 적용하여 network를 정규화하는데 도움을 줌

그림 4는 Feed-Forward Network(FFN) module을 나타냄

### 2.4. Conformer Block

제안한 conformer block에는 그림 1과 같이 **multi-head self-attention module과 convolution module 사이에 2개의 feed-forward module**이 포함됨  
- 이 샌드위치 구조는 transformer block의 원래 feed-forward layer를 2개의 half-step feed-forwar layer(attention layer 전 후로 배치)로 대체한 Macaron-Net[18]에서 영감을 얻었음
- Macron-Net에서와 같이 본 논문의 feed-forward layer에서 half-step residual weight를 사용함

두번째 feed-forward module 다음에 최종 layernorm layer가 옴

수학적으로 conformer block i에 대한 입력 $x_i$에 대해 block의 출력 $y_i$가 다음과 같다는 것을 의미함

$\tilde{x_i} = x_i + \frac{1}{2}FFN(x_i)$
$x'_i = \tilde{x_i} + MHSA(\tilde{x_i})$

$x''_i = x'_i + Conv(x'_i)$

$y_i = Layernorm(x''_i + \frac{1}{2}FFN(x''_i))$

section 3.4.3에서 이전 작업에서 사용된 **vanilla FFN과 Macron-style의 half-step FFN을 비교**함

- 2개의 macaron-net style feed-forward layer 사이에 attention module과 convolution module을 끼워넣는 half-step residual connection이 있는게 conformer architecture에서 단일 feed-forward module을 사용하는 것보다 **상당히 개선**된다는 것을 발견함

convolution과 self-attention의 조합은 이전에 연구되었으며 이를 달성하는 많은 방법을 상상할 수 있었음
self-attention으로 convolution을 증가시키는 다양한 옵션은 section 3.4.2에 작성

⇒ **self-attention module 뒤에 쌓인 convolution module**이 음성 인식에 가장 잘 작동하는 것을 발견

# *3. Experiments*

### 3.1 Data

970시간 labeled speech와 language model 구축을 위한 추가 800M word token text전용 corpus로 구성된 LibriSpeech dataset에서 제안된 모델을 평가
- 25ms window, 10ms stride
- 80-channel filterbank feature

SpecAugment [27, 28] with mask parameter (F=27)와 최대 time-mask ratio(ps=0.05)를 가진 10개 time mask 사용
- time msak의 최대 size는 발화 길이 * ps로 설정

### 3.2 Conformer Tranducer

network 깊이, model dimension, attention head 수의 다양한 조합을 스위핑하고, model parameter size 제약 내에서 가장 성능이 좋은 모델을 선택해 10M, 30M, 118M  parameter를 사용하여 소, 중, 대 세가지 모델을 식별
- 모든 모델에서 single-LSTM layer decoder를 사용

표 1은 architecture hyperparameter를 보여줌

<img src="https://user-images.githubusercontent.com/46676700/128827078-593e8915-0585-42e0-b603-f3974ec64f4d.png" alt="img" style="zoom:60%;"/>

- **dropout**: module 입력에 추가되기 전에 conformer의 각 residual unit, 즉 각 module의 출력에 적용 (비율 $P_{drop}$ = 0.1)
- **Variational noise**[5, 30]
- **L2 regularization**: 1e-6 weight (모든 학습 가능한 wight에 추가)
- **Adam** optimizer(β1 = 0.9, β2 = 0.98, ε = 10−9)
- **transformer** **learning rate schedule** (10k warm-up step, 최대 learning rate $\frac{0.05}{\sqrt{d}}$ (d: model dimension)
- **3-layer LSTM LM** (width 4096)
  - LibriSpeech 960h에서 구축된 1k Words Per Minute(WPM)으로 tokenized LibriSpeech960h transcript가 추가된 LibriSpeech language model corpus에서 훈련
  - LM은 dev-set transcript의 word-level perplexity(혼란도)가 63.9
  - shallow fusion에 대한 LM weigth λ는 grid search를 통해 dev-set에서 조정

⇒ 모든 모델은 **Lingvo toolkit**으로 구현

### 3.3 Results on LibriSpeech

<img src="https://user-images.githubusercontent.com/46676700/128827091-238c5479-203d-4918-b555-655df0c6614a.png" alt="img" style="zoom:60%;"/>

표 2는 LibriSpeech test-clean/test-other에 대한 모델의 WER 결과를 ContextNet, Transformer transducer 및 QuartzNet을 포함한 몇 가지 최신 모델과 비교
- 모든 평가 결과는 소수점 이하 1자리로 반올림

**언어 모델 X**  
- 중간 모델의 성능은 test/testother에서 이미 가장 잘 알려진 Transformer, LSTM 기반 모델 또는 유사한 크기의 convolution 모델을 능가하는 2.3/5.0로 경쟁력 있는 결과를 달성

**언어 모델 O**  
- 모든 기존 모델 중 가장 낮은 WER
- single NN에서 Transformer와 convolution을 결합하는 것의 효율성을 분명히 보여줌

### 3.4 Ablation Studies

> ***3.4.1. Conformer Block vs Transformer Block***

Conformer block은 여러 방면에서 Transformer block과 다름

특히, macaron-style의 convolution block과 이를 둘러싼 FFN pair가 존재
⇒ 총 parameter 수를 변경하지 않고, conformer block을 transformer block으로 변경하여 차이를 확인

표 3는 conformer block에 대한 각 변형의 영향을 나타냄

<img src="https://user-images.githubusercontent.com/46676700/128827098-dde8d71e-599e-405c-b83e-ad70b5fe9e0e.png" alt="img" style="zoom:60%;"/>

모든 차이점 중에서 **convolution sub-block**이 가장 중요한 feature이지만 macaron-style의 FFN pair를 갖는 것이 동일한 수의 parameter를 갖는 single FFN보다 더 효과적

swish activation을 사용하면 Conformer 모델에서 더 빠른 수렴이 이루어짐

> ***3.4.2 Combinations of Convolution and Transformer Modules***

MHSA module과 convolution module을 결합하는 다양한 방법의 효과를 연구

1. convolution module의 depthwise convolution을 lightweight convolution[35]으로 교체 시도
- 특히, dev-other dataset에서 성능이 크게 떨어지는 것을 볼 수 있음
1. Conformer 모델에서 MHSA module 앞에 convolution module을 배치
- dev-other에서 0.1만큼 결과가 저하시키는 것을 발견
1. [17]에서 제안한 대로 output이 연결된 multi-head self-attention module과 convolution module의 parallel branch로 input을 분할
- 제안한 architecture와 비교할 때 성능을 악화시킨다는 것을 발견

⇒ 표 4는 Conformer block에서 self-attention module 뒤에 convolution module을 배치하는 이점을 시사함

<img src="https://user-images.githubusercontent.com/46676700/128826577-dfee6a64-4c88-426e-8652-9ea83a8a39de.png" alt="img" style="zoom:60%;"/>

> ***3.4.3. Macaron Feed Forward Modules***

Transformer 모델에서와 같이 attention block 이후 single FFN 대신 Conformer block에는 self-attention 및 convolution module 사이에 macaron과 같은 한 쌍의 feed-forward module이 있음

또한, Conformer feed-forward module은 half-step residule과 함께 사용됨

표 5는 single FFN 또는 전체 full-step residual을 사용해 Conformer block을 변경할 때 결과를 나타냄  
- 차이가 많이 없지만, macaron style feed-forward module이 가장 좋은 성능을 보임

<img src="https://user-images.githubusercontent.com/46676700/128827103-fde4055b-51d1-48d7-a372-6e8e0624c306.png" alt="img" style="zoom:60%;"/>

> ***3.4.4. Number of Attention Heads***

self-attention에서 각 attention head는 입력의 다른 부분에 초점을 맞추어 학습하여 단순한 weighted average 이상으로 predict를 개선할 수 있음

large 모델에서 모든 layer에서 4~32까지 동일한 수의 attention head를 변경하면서 사용해 효과를 연구하기 위해 실험을 수행

표 6에서 볼 수 있듯이 특히 dev-other dataset에 대해 attention head를 최대 16까지 증가시키면 정확도가 향상된다는 것을 발견

<img src="https://user-images.githubusercontent.com/46676700/128826587-7f85fcb1-4ec0-4660-8df9-5d54144b5562.png" alt="img" style="zoom:60%;"/>

> ***3.4.5. Ablation study on depthwise convolution kernel sizes***

depthwise convolution에서 kernel size의 영향을 연구하기 위해 모든 layer에 대해 동일한 kernel size를 사용해 large 모델에서 kernel size를 {3, 7, 17, 32, 65}로 스윕하여 실험

kernel size 17과 32까지 size가 클수록 성능이 향상되지만, 표 7에서 볼 수 있듯이 size 65의 경우에는 성능이 악화된다는 것을 발견

dev WER에서 소수 둘째자리를 비교하면 비교하면 나머지보다 size 32가 더 나은 성능을 보임

<img src="https://user-images.githubusercontent.com/46676700/128826596-fede2ba9-ac3d-4a18-b5ac-0c2a6196592e.png" alt="img" style="zoom:60%;"/>

# *4. Conclusion*

본 몬문에서는 End-to-End speech recognition을 위해 **CNN 및 Transformer의 구성 요소를 통합**하는 architecture인 **Conformer를 도입**

각 구성 요소의 중요성을 연구해 Convolution module을 포함하는 것이 Conformer 성능에 중요하다는 것을 보여줌

LibriSpeech dataset에 대한 이전 model보다 더 적은 parameter로 향상된 정확도를 보임  
- **test/test-other에 대해 1.9%/3.9%로 SOTA 달성**

---

### **Further reading**
**💡 translation equivariance**  
[What is translation equivariance, and why do we use convolutions to get it?](https://chriswolfvision.medium.com/what-is-translation-equivariance-and-why-do-we-use-convolutions-to-get-it-6f18139d4c59)

**💡 Transformer와 구조적으로 비교**  
[kakaobrain/nl-paper-reading](https://github.com/kakaobrain/nlp-paper-reading/blob/master/notes/conformer.md)
