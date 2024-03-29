---
layout: post
title: "Text-Independent Speaker Verification with Adversarial Learning on Short Utterances"
date: 2019-07-24
category: review
thumbnail: /style/image/GAN_shortutt.png
use_math: true
icon: book
---

* contents
{:toc}

<span style="font-size:13pt">Kai Liu, Huan Zhou</span>

# 📌 **Abstract**

**문제점:** Text-independent speaker verification은 짧은 발화 조건에서 심각한 성능 저하를 겪음
**해결방법:** short embedding을 enhanced embedding에 직접 매핑하여 판별력(discriminability)을 높이도록 adversarial하게 훈련된 embedding model 제안  

- 특히, loss criteria(기준)이 많은 <span style="background-color:#AED6F1">**Wasserstein GAN**</span> 사용
- 여러 loss function은 뚜렷하게 최적화하려는 목표를 가지고 있으나 그 중 일부는 화자 검증 연구에 도움이 되지 않음
- 대부분의 이전 연구와 달리  <span style="background-color:#AED6F1">**이 연구의 주요 목표** 는 **수많은 ablation 연구** 로 부터 loss criteria의 효과를 검증</span>
　→ 위에서 말하는 SV에서 도움이 되지 않는 loss들을 제거하면서 loss에 따른 영향을 조사
- VoxCeleb dataset에 대한 실험에서 일부 criteria는 SV 성능에 이로운 반면 일부 criteria는 사소한 영향을 미친다는 것을 보여줌
- 마지막으로, finetuning없이 사용한 Wasserstein GAN은 baseline을 넘어 의미 있는 성능 향상을 달성하며, EER에서는 4%의 상대적 개선과 2초간의 짧은 발화의 challenge한 시나리오에서는 7%의 minDCF를 달성


---


# **Ⅰ. Introduction** 🌱

- TI-SV: 등록된 화자와 테스트 음성(내용 제약 X)을 통해 화자의 신원을 검증
- 중요한 단계: 임의의 지속시간을 갖는 음성을 고정 차원의 speaker representation으로 매핑하는 것 (acoustic feature → speaker feature)
- Baseline System: GhostVLAD-aggregated embedding(G-vector); 긴 발화, 짧은 발화에서 좋은 성능을 보였으며, 잡음 환경에서 x-vector보다 이점이 있어 SV 시스템에 더 유리
- NIST-SRE 2010 test set에서 **full-duration이 5초로 단축**되었을 때 i-vector/PLDA system **성능이 2.48%에서 24.78%** 로 감소, **최근 딥러닝 기술 사용하여 이를 보완하는 연구가 많이 진행 중**
- 본 논문에서는 Wasserstein GAN의 adversarial 학습을 이용하여 향상된 차별성을 가진 새로운 embedding을 제안
(같은 화자의 짧은 발화와 긴 발화에서 추출한 G-vector를 활용하여)


---


# **Ⅱ. Related Work** 🌿

**✔ GAN 이란**: 생성자(Generator)와 식별자(Discriminator)가 싸우면서 학습하는 모델
- Generator : Discriminator를 속이도록 학습
- Discriminator : real sample 𝑦와 noise 𝜂로부터 생성된 fake sample 𝑔의 차이를 학습

</br>

**✔ Adversarial Learning**
- minmax loss function이 교대로 최적화 과정을 수행 (두 모델의 loss가 같아지는 상태가 될 때까지)

<center><img src="https://user-images.githubusercontent.com/46676700/101442311-735b3b80-395e-11eb-87da-130ab93a5834.png" alt="img"/></center>

- Gradients diminishing, exploding 문제로 훈련하기 어려운데 이를 Wasserstein GAN(WGAN)에서 수학적으로 다루었음
- Discriminator는 좋은 $𝑓_𝑤$를 찾도록 설계되었으며, 새로운 loss function은 Wasserstein 거리를 측정하도록 구성


<center><img src="https://user-images.githubusercontent.com/46676700/101442322-76eec280-395e-11eb-8f23-77c965f91d6a.png" alt="img"/></center>


---


# **Ⅲ. Proposed Approach** 🌳

- 제안하는 전급 방식은 아래의 구조와 같음

<center><img src="https://user-images.githubusercontent.com/46676700/101442513-ccc36a80-395e-11eb-923b-4ca1aa2ac183.png" alt="img"/></center>

> $𝑥, 𝑦$ : 같은 speaker의 각각 짧고 긴 발화에 해당하는 D차원의 G-vector  
> $𝑧$ : speaker ID label  
> $𝐺_𝑓$ : embedding generator  
> $𝐺_𝑐$ : speaker label predictor  
> $𝐺_𝑑$ : Distance calculator  
> $𝐷_𝑤$ : Wasserstein discriminator  

<br/>

- 제안된 방법의 **핵심적인 task**는 **discriminability이 향상된 embedding을 학습**하는 것

<span style="background-color:#E4C4F0">**✔ loss functions**</span>

- **WGAN loss**
<center><img src="https://user-images.githubusercontent.com/46676700/101443118-02b51e80-3960-11eb-86ce-40b44aed35fc.png" alt="img"/></center>

<br/>

- **Conditional WGAN loss**: GAN에 Wasserstein 거리를 이용한 새로운 loss function 정의

  - $𝑥$ (짧은 발화 embedding)이 주어졌을 때, $𝐷_𝑤$와 $𝐺_𝑓$ 분포의 차이 ($𝑥$와 real sample, fake sample을 연결하여 학습)


<center><img src="https://user-images.githubusercontent.com/46676700/101443121-047ee200-3960-11eb-85e3-d6cdb120eb2f.png" alt="img"/></center>

<br/>

⚡️ WGAGN loss / Conditional WGAN loss 중 하나만 사용하고, 그 차이를 성능 평가 실시

</br>

- **FID loss**: Fréchet Inception Distance

  - Real sample과 fake sample의 벡터 사이의 거리 계산을 위한 metric

<center><img src="https://user-images.githubusercontent.com/46676700/101443125-05b00f00-3960-11eb-83a1-b11abcfe6840.png" alt="img"/></center>

<br/>

- **class loss**: Multi-class cross-entropy loss

  - Speaker에 따른 embedding 차이를 위한 loss 정의

<center><img src="https://user-images.githubusercontent.com/46676700/101443129-08126900-3960-11eb-98d6-16200989d2ff.png" alt="img"/></center>

> $𝑁$ : Batch size  
> $𝑐$ : Class 수  
> $𝑔_𝑖$ : i번째 생성된 embedding  
> $𝑧_𝑖$ : 해당 label index  
> $𝑊∈ℜ^(𝐷∗𝑐), 𝑏∈ℜ^𝑐$ : weight matrix, bias  


<br/>

- **Triplet loss**

  - Class 분류 시 error에 대한 패널티

<center><img src="https://user-images.githubusercontent.com/46676700/101443133-09dc2c80-3960-11eb-882f-caf3570671b9.png" alt="img"/></center>

> $\Gamma$ : training set에서 가능한 모든 embedding의 triplet $\gamma=(𝑔_𝑎, 𝑔_𝑝, 𝑔_𝑛)$의 set  
> $𝑔_𝑎$ : anchor input  
> $𝑔_𝑝$ : positive input  
> $𝑔_𝑛$ : negative input  
> $\Psi∈ℜ^+$ : positive와 negative 사이의 safety margin  


<br/>

- **Center loss**

  - Class 내 variation 최소화

<center><img src="https://user-images.githubusercontent.com/46676700/101443140-0c3e8680-3960-11eb-8896-37b5be81d367.png" alt="img"/></center>

> $𝑐_(𝑦_𝑖)$ : deep feature의 𝑦_𝑖번째 class center  
> $𝑥_𝑖$ : $𝑦_𝑖$번째 class에 속하는 𝑖번째 deep feature  
> $𝑚$ : mini-batch size  


<br/>

- **Cosine distance loss**

  - Generator model로 얻은 향상된 embedding과 real sample(target) 사이의 유사도를 고려

<center><img src="https://user-images.githubusercontent.com/46676700/101443144-0ea0e080-3960-11eb-8437-04997a2f26bc.png" alt="img"/></center>

> $\bar 𝑒$: normalized embedding

<br/>

:star: <span style="background-color:#FFED81">**✔ Generator와 Discriminator의 최종 Loss**</span>

- $G_f$

<center><img src="https://user-images.githubusercontent.com/46676700/101444427-d3ec7780-3962-11eb-8967-3f0ff2912fad.png" alt="img"/></center>  

- $D_w$

<center><img src="https://user-images.githubusercontent.com/46676700/101444430-d5b63b00-3962-11eb-97a0-807326d8a4a4.png" alt="img"/></center>  

- WGAN 훈련 후 generative model $𝐺_𝑓$ 유지

  - Test 단계에서 짧은 발화 embedding $𝑥$를 $𝐺_𝑓$에 넣어 enhanced embedding($g$)를 얻음


---

# **Ⅳ. Experiments and Results** 🌺

**✔ Experimental setup**

- **Train:**  VoxCeleb2의 subset (1,057명 화자의 164,716개 발화)
- **Test:**   VoxCeleb1의 subset (40명 화자의 13,265개 발화)
- 짧은 발화를 위해 **random하게 2초 잘라서** 사용

**✔ Baseline system**

- G-vector (VGG-Restnet34s)

**✔ Hyper Parameter**

- Learning rate 0.0001
- Adam Optimizer
- Weight clipping -0.01 ~ 0.01 threshold ($𝐷_𝑤$)
- Batch size 128

<br/>

<span style="background-color:#AED6F1">**✔ 다양한 loss function의 영향 연구**</span>

<center><img src="https://user-images.githubusercontent.com/46676700/101445011-0d71b280-3964-11eb-8d77-389d6aa37ee3.png" alt="img"/></center>
<center><img src="https://user-images.githubusercontent.com/46676700/101445016-0f3b7600-3964-11eb-94f9-767587338bcc.png" alt="img"/></center>  

<center> - FID loss은 긍정적인 영향 (v1 vs v2) </center>
<center> - Conditional WGAN이 WGAN보다 나음 (v3 vs v4) </center>
<center> - Triplet loss를 넣으면 조금 더 나은 결과를 보임 (v7 vs v2) </center>
<center> - Triplet b(fake)보다 Triplet a(real, fake 모두)가 크게 성능 향상 (v3 vs v8) </center>
<center> - Softmax는 긍정적인 영향 (v3 vs v5) </center>
<center> - Center loss은 부정적인 영향 (v6 vs v7) </center>
<center> - Cosine loss은 긍정적 영향 (v6 vs v8) </center>

<br/>

- **추가적인 training function**(softmax, cosine, triplet)이 모두 **훈련에 긍정적인** 영향을 미침
- SV시스템에 FID, conditional WGAN은 매우 유용, 추가 조사 가치가 있음

<br/>

**✔ Baseline system과 비교**

- 실험 중 가장 성능이 좋았던 v3 system과 G-vector baseline system 비교
  - EER과 minDCF

<center><img src="https://user-images.githubusercontent.com/46676700/101445333-a6083280-3964-11eb-86af-900deb097f6e.png" alt="img"/></center>

<br/>

- Baseline보다 짧은 duration에 대해 더 나은 성능을 보임
  - 상대적으로 EER은 4.2% 개선하였으며, minDCF는 7.2% 개선 – 1초 task에서도 상대적 EER 3.8% 향상
- 시간 제약으로 FID loss는 최종 system에 추가하지 않았으며 hyper-parameter, loss weight($\alpha, \beta, \gamma, \lambda, \epsilon$)와 triplet margin $\Psi$에 대한 미세조정이 없었음
  - 제안한 system의 개선될 여지가 많이 남아있음

---

# **Ⅴ. Conclusion** 🌞

- 본 논문에서는 **WGAN을 적용** 하여 **발화가 짧은** speaker verification application의 **향상된 embedding을 성공적으로 학습**
- 제안된 WGAN 기반 커널 시스템 그리고 그 위에, GAN 훈련에서 **많은 loss criteria의 효과를 검증**
- 최종 제안 시스템은 도전적인 짧은 스피커 검증 시나리오에서 baseline system을 능가
- 전반적으로, 상당한 진보와 연구가 진전되는 잠재적 방향을 보여줌
