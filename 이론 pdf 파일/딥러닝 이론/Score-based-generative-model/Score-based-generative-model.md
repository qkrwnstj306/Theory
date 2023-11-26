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
- 존재하는 생성 모델은 확률 분포를 표현하는 방법에 따라 두 category 로 분류할 수 있다.
    1. Likelihood-based models
        - 분포의 PDF 를 (approximate) maximum likelihood 를 통해 직접적으로 배운다. 
        - E.g., 
          - Autoregressive models 
          - Normalizing flow models
          - Energy-based models (EBMs)
          - VAE
    2. Implicit generative models 
        - Sampling process 로 내재적으로 확률분포를 표현한다.
        - E.g.,    
          - GAN

- 하지만, 이 두 가지 방법론은 significant limitation 이 존재한다. 
  - Likelihood-based models: tractable normalizing constant 를 보장해야 해서 model 구조의 강력한 제한이 있거나 / maximum likelihood 를 근사하는 objective 로 설정 (ELBO of VAE)
  - Implicit generative models: Adversarial training -> Unstable, mode collapse

- 본 post 에선, 이런 제한을 우회하면서 확률 분포를 표현하는 다른 방법을 소개한다.

$$ \nabla_x \log{p(x)} = Score \ function $$

- Normalizing constant 가 tractable 하지 않아도 된다.
- Score matching 을 통해, 확률 분포를 직접 배운다. 


***

### <strong>Method</strong>
- Dataset $\{x_1,x_2, \cdots, x_N\}$ 이 주어졌을 때, model 이 $p(x)$ 를 알기를 원한다. 
  - $p(x)$ 를 먼저 표현할 줄 알아서 model 을 통해 근사시킬 수 있다.
- Score fucntion
- Score-based models
- Score matching

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