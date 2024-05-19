<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## NICE: Non-Linear Independent Components Estimation
*arXiv(2014), 2221 citation, Review Data: 2024.05.18*

[Intro](#intro)</br>
[Related Work](#related-work)</br>
[Method](#method)</br>
[Experiment](#experiment)</br>
[Conclusion](#conclusion)</br>

<p align="center">
<img src='./img1.png'>
</p>

> Core Idea
<div align=center>
<strong>"test1"</strong></br>
</div>

***

### <strong>Intro</strong>


<p align="center">
<img src='./img2.png'>
</p>

$\textbf{이 주제의 정의 및 요구사항과 중요한 이유}$

- 알 수 없는 복잡한 data distribution 을 modeling 하는 것은 중요하다. 


$\textbf{이 주제의 문제점과 기존의 노력들}$

- 대부분의 딥러닝 모델에서 사용하는 함수들은 역함수가 존재하지 않는다. (ReLU, Pooling, Convolution)
- Jacobian 의 determinant 를 구하기 어렵다 (계산량 등)

$\textbf{본 논문에서 해결하고자 하는 문제와 어떻게 해결하는지, 그 결과들}$

- Motivation: data 의 분포가 modeling 되기 쉽게 바꾸자 using transformation 

- 복잡한 high-dimensional densities 를 modeling 하는 deep learning framework 를 제안한다.
  - Latent space 로 data 를 mapping 하는 non-linear deterministic transformation $f$ 를 학습한다. 
  - Transformed data 는 독립적인 latent variable 을 가지는 분포이다. (즉, 우리가 modeling 하기 쉬운 분포로 변환)
  - 본 논문에서는 Jacobian 과 역 Jacobian 의 determinant 계산이 쉬우면서도 복잡한 non-linear transformation 을 학습할 수 있도록 이 transformation 을 parameterize 했다. 
  - 학습은 정확한 log-likelihood (=tractable) 이다.
  - Generative model: $4$ 가지 image dataset (can be used for inpainting)

$\textbf{본 논문의 주요 기여점}$

***

### <strong>Related Work</strong>


***

### <strong>Method</strong>

먼저 *Normalizing Flow* 의 기본 이론에 대해서 알아보자!!

- Inference: data $x$ 를 $h$ 로 mapping
- Generation (sampling): $h$ 에서 $x$ 로 mapping 

<p align="center">
<img src='./img4.png'>
</p>

$\textbf{Change of Variable}$

- $x \sim p_X(x)$: data (random variable)
- $h \sim p_H(h)$: random variable and its known pdf (in this paper, prior distirbution $p_{H_N}$ 는 standard isotropic Gaussian)
- $f$: invertible transformation 
- $h = f(x), x = f^{-1}(h)$


- 여기서 알 수 있는건, 어떤 비선형 변환 $f$ 를 data $x$ 에 적용시켜 $h$ 를 만들었다는 것이다.  

- PDF 의 적분은 $1$ 임을 이용하여 전개한다.

$$ \int p_X(x) dx =  \int p_H(h) dh =1 $$

$$ \frac{d}{d x} \int p_X(x) dx = \frac{d}{d x} \int p_H(h) dh $$

$$ p_X(x) = \frac{d}{d x} \int p_H(h) \det{(\frac{dh}{dx})} dx = p_H(h) \det{(\frac{dh}{dx})}= p_H(f(x))\det(\frac{df(x)}{dx}) $$

- 결국엔 다음의 수식을 만족한다. 

$$ p_X(x) = p_H(f(x))\det(\frac{df(x)}{dx}) $$

- **NOTE**
  - $x$ 에 어떤 비선형 변환 $f$ 을 적용하여 $h$ 에 mapping 한다고 봤을때, 이는 미소 영역에 대해서 $x$ 와 Jacobian 행렬을 곱해서 선형 변환으로도 볼 수가 있다. 
  - 이때, Jacobian 의 determinant 는 $x$ 기저가 이루는 단위 면적이 $h$ 로 변환하면 얼마나 늘어나는 가를 나타낸다. 

$$ dh = \det{(\frac{dh}{dx})} dx = \det(\frac{df(x)}{dx})dx$$

- 보통 $x$ 는 매우 복잡한 분포를 갖는다. 따라서 $x$ 와 $z$ 는 하나의 함수로 연결하기는 어려워서 많은 함수를 이어서 연결해준다. 
  - 즉, DDPM 과 다르게 한 layer 가 transformation $f$ 을 담당한다. 

<p align="center">
<img src='./img3.png'>
</p>

$$
p_X(x) = p_{H_1}(h_1) \det(\frac{dh_1}{dx}) \\

p_{H_1}(h_1) = p_{H_2}(h_2) \det(\frac{dh_2}{dh_1}) \\

p_{H_2}(h_2) = p_{H_3}(h_3) \det(\frac{dh_3}{dh_2}) \\

. \\ 
. \\
. \\ 

p_{H_{N-1}}(h_{N-1}) = p_{H_N}(h_N) \det(\frac{dh_N}{dh_{N-1}}) \\

$$

- So, 

$$ p_X(x) = p_{H_N}(h_N) \prod_{i=1}^N\det(\frac{dh_i}{dh_{i-1}}) \ , \text{if} \ \ i=1, \ h_0 = x $$

- $\log$ (최종적인 목적 함수)

$$ \log p_X(x) = \log p_{H_N}(h_N) + \sum_{i=1}^N\log \det(\frac{dh_i}{dh_{i-1}}) \ , \text{if} \ \ i=1, \ h_0 = x $$

- 여기서 우리는 *normalizing flow* 를 구현하기 위해 $2$ 가지 조건이 필요함을 알 수 있다. 
  1. Transformation $f$ 의 역함수가 존재해야한다. 
  2. Jacobian 의 determinant 를 계산할 수 있어야 한다. 

$\textbf{Coupling Layer}$

- 본 논문의 핵심은 Jacobian 의 determinant 와 inverse $f^{-1}$ 을 쉽게 구할 수 있게 설계하는 것이다. 
  - 그 첫번째가 coupling layer 를 사용하여 $f$ 를 구성하는 것이다. 

- $x$ 를 $(x_1, x_2)$ 로 나누고 $(y_1, y_2)$ 의 form 으로 변환한다. 
  - $m$: 임의의 복잡한 함수 (E.g., ReLU MLP)
  - 즉 $m$ 에서만 neural network 가 존재한다. 

- 일반적인 coupling layer 

<p align="center">
<img src='./img5.png'>
</p>

<p align="center">
<img src='./img6.png'>
</p>

- 일반적인 coupling layer 에서, $g$ 를 additive coupling layer 로 본다면 다음과 같다.

$$ y_1 = x_1 \\ y_2 = x_2 + m(x_1) $$

- 이렇게 변환하면 어떤 $m$ 에 대해서도 Jacobian determinant 가 $1$ 이고 역함수도 다음과 같이 구하기 쉽다. 
  - $\frac{\partial y_{I_2}}{\partial x_{I_2}} = 1$
  - Triangular matrix 의 determinant 는 주대각 성분들의 곱으로 표현된다. 


$$ x_1 = y_1 \\ x_2 = y_2 - m(y_1) $$


$\textbf{Code Reivew}$

- Sampling 
  - $2$ 개의 prior distribution 중 선택해야한다. 이는 논문에도 나와있다. 
  - Input dim 의 size 로 생성할건데, grid size 는 args.nrows 와 args.ncols 로 결졍된다. 
  - nice.inverse 를 통해 sampling 된다. 

<p align="center">
<img src='./img8.png'>
</p>

<p align="center">
<img src='./img7.png'>
</p>

- Loss 

<p align="center">
<img src='./img9.png'>
</p>

<p align="center">
<img src='./img10.png'>
</p>

<p align="center">
<img src='./img11.png'>
</p>

- Coupling layer
  - 항상 같은 위치의 $x_1$ 이 network input 으로 들어가는게아니라 'even', 'odd' 로 구분지어서 짝수 인덱스 열 또는 홀수 인덱스 열에 대해서 $x_1, x_2$ 가 선정된다.
  - Partition 이 even 이면 홀수 인덱스 열이 $x_1$ 이 되고 layer 의 input 으로 들어가서 수식이 전개된다.
  - 각 layer 는 ReLU 로 비선형성을 챙겼다.
  - Inverse 는 forward 의 반대순서로 간다. 
  - _get_even: 짝수 인덱스만 챙긴다.

<p align="center">
<img src='./img12.png'>
</p>

<p align="center">
<img src='./img13.png'>
</p>

<p align="center">
<img src='./img15.png'>
</p>

<p align="center">
<img src='./img17.png'>
</p>

<p align="center">
<img src='./img18.png'>
</p>


<p align="center">
<img src='./img19.png'>
</p>

<p align="center">
<img src='./img20.png'>
</p>


***

### <strong>Experiment</strong>

<p align="center">
<img src='./img21.png'>
</p>

<p align="center">
<img src='./img22.png'>
</p>

***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>


