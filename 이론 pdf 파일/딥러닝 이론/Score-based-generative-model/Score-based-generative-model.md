![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Theory%20Review&fontSize=50)


## Score-Based Generative Models
*A representative reference: <a href='https://yang-song.net/blog/2021/score/'>Yang Song blog</a>*

[Intro](#intro)</br>
[Related Work](#related-work)</br>
[Method](#method)</br>
[Experiment](#experiment)</br>
[Conclusion](#conclusion)</br>

> Core Idea

$$ utilize \ \nabla_x \log{p(x)} $$

***

### <strong>Intro</strong>



***

### <strong>Related Work</strong>

$$ p(x) = \frac{p(x|y)p(y)}{p(y|x)} $$

- 존재하는 생성 모델은 확률 분포를 표현하는 방법에 따라 두 category 로 분류할 수 있다.
    1. Likelihood-based models
        - 분포의 PDF/PMF 를 (approximate) maximum likelihood 를 통해 직접적으로 학습한다. 
        - E.g., 
          - Autoregressive models (PixelRNN, etc.)
          - Normalizing flow models
          - Energy-based models (EBMs)
          - VAE: 
            - Bayes rule 로 표현한 값의 사후 확률까지 approximation 하고자 했다.
            - ELBO trick 으로 확률 분포를 근사한다.  
    2. Implicit generative models 
        - 확률 분포를 sampling process 로 암시적으로 나타낸다. 
        - E.g.,    
          - GAN: 확률 분포를 직접 근사하지는 않고, 다른 loss 를 이용한다. 이때 이 loss 를 푸는 과정이 암시적으로는 확률 분포를 푸는 문제와 동일하다고 볼 수 있다. 즉, 간접적으로 푸는 형태.

<p align="center">
<img src='./img1.png'>
</p>

- 하지만, 이 두 가지 방법론은 significant limitation 이 존재한다. 
  - Likelihood-based models
    - Tractable normalizing constant 를 보장해야 해서 model 구조의 강력한 제한이 있거나
      - E.g., causal convolutions in autoregressive models, invertible networks in normalizing flow models
    - Approximate the normalizing constant -> 계산 비용이 많이 든다.
      - E.g., variational inference in VAEs, MCMC sampling used in contrastive divergence
    - Maximum likelihood 를 근사하는 objective 로 설정. 정확한 계산이 아니다.
      - E.g., ELBO of VAE
  - Implicit generative models
    - Adversarial training: Unstable, mode collapse

- 본 post 에선, 이런 제한을 우회하면서 확률 분포를 표현하는 다른 방법을 소개한다.
    - Normalizing constant 가 tractable 하지 않아도 된다.
    - Score matching 을 통해, 확률 분포를 직접 배운다. 

$$ \nabla_x \log{p(x)} = Score \ function $$

<p align="center">
<img src='./img2.png'>
</p>

> Normalizing constant
>> 정규화 상수의 개념은 확률 이론 및 기타 영역에서 다양하게 발생한다. 정규화 상수는 확률 함수를 전체 확률이 $1$인 확률 밀도 함수로 변환하는 데 사용된다. 

***

### <strong>Method</strong>
#### Score fucntion
  
- Dataset $\{x_1,x_2, \cdots, x_N\}$ 이 주어졌을 때, model 이 $p(x)$ 를 알기를 원한다. 
  - $p(x)$ 를 먼저 표현할 줄 알아야 model 을 통해 근사시킬 수 있다.

$$ p_\theta(x) = \frac{e^{-f_\theta(x)}}{Z_\theta} , \ Let \ f_\theta(x) \ is \ scalar, \ learnable\ parameter\  \theta $$

$$ Z_\theta > 0, \ is \ a \ normalizing \ constant \ dependent \ on \ \theta, \ such \ that \int{p_\theta(x)dx = 1} $$

> $f_\theta(x)$ 는 unnormalized probabilistic model or energy-based model 이라고 부른다. 
- 따라서, maximizing log-likelihood of the data 를 통해 $p_\theta(x)$ 를 학습할 수 있다.

$$ max_\theta \Sigma_{i=1}^{N}{\log{p_\theta(x_i)}} $$

- 하지만, log-likelihood 를 maximize 하기 위해선 $p_\theta(x)$ 가 정규화된 PDF(*normalized probability density function*)여야 한다. 이건 계산상의 어려움을 일으키는데, 우리는 일반적인 $\theta$ 에 대해 일반적으로 복잡한 양인 정규화 상수 $Z_\theta$ 를 계산해야 하기 때문이다.
- 따라서, maximum likelihood 를 가능하게 하려면 likelihood-based model 은 모델 구조를 제한하거나 정규화 상수를 다루기 쉽게 만들기 위해 정규화 상수를 근사해야 했다. 
- Density function 대신 Score function 을 modeling 함으로써, 우리는 intractable normalizing constant 의 어려움을 피할 수 있다. 

> Score function 에 대한 model 을 **Score-based model** 이라고 부른다.

- Score-based model $s_\theta(x)$ 는 Score fucntion 을 학습한다. 
    - $s_\theta(x)$ 가 normalizing constant $Z_\theta$ 에 독립적이라는 것에 주목해야 한다.
    - Normalizing constant tractable 을 보장하기 위해 특별한 모델 구조를 설계할 필요가 없다.
<p align="center">
<img src='./img3.png'>
</p>

- 즉 loss 는 다음과 같이 정의된다. using Fisher divergence.

<p align="center">
<img src='./img4.png'>
</p>

- 직접적으로 divergence 를 계산하려고 했지만, $\nabla_x \log{p(x)}$ 를 알지 못한다.
  - 다행히, score matching method 를 통해 ground-truth data score 를 몰라도 Fisher divergence 를 minimize 할 수 있다.

#### Score matching

#### Sampling
Langevin dynamics 
- Score-based model $s_\theta(x) \approx \nabla_x \log{p(x)}$ 을 학습했으면, Langevin dynamics 라고 불리는 iterative procedure 을 통해 sampling 을 하면 된다. 
- Langevin dynamics 는 오직 score function $\nabla_x \log{p(x)}$ 만을 사용해, $p(x)$ 로부터 MCMC procedure 를 제공한다.  
- 구체적으로, 이 방법은 arbitrary prior distribution $x_0 \sim \pi(x)$ 에서 다음을 반복한다.

<p align="center">
<img src='./img5.png'>
</p>


#### asd
- 지금까지는 score matching 을 사용하여, score-based model 을 훈련하고 Langevin dynamics 를 통해 sampling 을 하는 방법을 살펴봤다. 그러나 이러한 단순한 접근 방식은 실제로는 제한된 성공을 거뒀다. 
- 이제는 score matching 의 몇 가지 문제들에 대해 얘기를 해본다.

<p align="center">
<img src='./img6.png'>
</p>

- 주요 문제는 적은 데이터 포인트가 존재하는 낮은 밀도 영역에서 추정된 score function 이 부정확하다는 사실이다. 이는 score matching 이 Fisher divergence 를 최소화하도록 설계되었기 때문에 예상된 결과라고 볼 수 있다.  
  - Langevin dynamics 로 sampling process 를 시작할 때, 초기 sample 은 높은 확률로 low density region 에 위치한다. 따라서 부정확한 score-based model 로 인해, sampling 이 올바른 방향으로 진행되지 않는다. 

<p align="center">
<img src='./img7.png'>
</p>

- 낮은 데이터 밀도 지역에서 정확한 score matching  의 어려움을 우회하는 해결책으로 데이터 포인트에 noise 를 적용하고 noise 가 추가된 데이터 포인트에서 score-based model 을 훈력하는 것을 제시한다. 
- Noise 의 크기가 충분히 큰 경우, 낮은 데이터 밀도 지역에 데이터를 채워 넣어 estimated score 의 정확도를 향상시킬 수 있다. 
- 그럼 우리가 생각해야 될 것은, 적절한 noise 크기를 어떻게 선택할 것인가이다. 큰 노이즈는 분명히 더 많은 낮은 밀도 영역을 포함하여 더 나은 score 를 추정할 수 있지만, 데이터를 지나치게 손상시키고 원래 분포에서 상당히 벗어날 수 있다. 반면 작은 노이즈는 원래 데이터 분포를 적게 손상시키지만 우리가 원하는 만큼 낮은 밀도 영역을 충분히 커버하지 못할 수 있다.  

<p align="center">
<img src='./img8.png'>
</p>

- 따라서, multiple scaled of noise perturbations 를 제안한다.

<p align="center">
<img src='./img9.png'>
</p>

- 각 noise-perturbed distribution $\nabla_x \log{p_{\sigma_i}(x)}$ 를 예측하면 된다. 
- 이때의, model 은 *Noise Conditional Score-Based Model* $s_\theta(x,i)$ 로써, NCSN 이라고 부른다. 

$$ s_\theta(x,i) \approx \nabla_x \log{p_{\sigma_i}(x)} , \ for \ all \ i=1,2, \cdots , L$$

<p align="center">
<img src='./img10.png'>
</p>

<p align="center">
<img src='./img11.png'>
</p>

<p align="center">
<img src='./img12.png'>
</p>

<p align="center">
<img src='./img13.png'>
</p>

#### Score-based models

***

### <strong>Experiment</strong>


***

### <strong>Conclusion</strong>
- Advangtage of Score-based generative models
  - GAN-level sample without adversarial training.
  - Flexibel model architecture.
  - Exact log-likelihood computation.
  - Inverse problem solving without re-training models.
- Process
    1. Large number of noise-perturbed data distributions
    2. Learn score function <score-matching>
    3. Samples with Langevin-type sampling 

***

### <strong>Question</strong>