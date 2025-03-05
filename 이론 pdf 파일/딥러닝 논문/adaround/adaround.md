<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## AdaRound: Up or Down? Adaptive Rounding for Post-Training Quantization
*International conference on machine learning. PMLR(2020), 597 citation, Qualcomm AI, Review Data: 2024.03.03*

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
  
- Neural network를 quantizing할 때, 각 floating-point weight를 nearest fixed-point value로 할당하는 것이 널리 퍼진 방법론이다. 
- 본 논문은 nearest rounding이 최선이 아니라, 여기서 제안한 **AdaRound**가 **post-training quantization**에서의 더 나은 weight-rounding mechanism이라고 주장한다.
- AdaRound는 빠르고 network의 추가적인 학습을 필요로하지 않는다. 오직 적은 양의 unlabelled data만을 사용한다. 
- 본 논문은 먼저 이론적으로 rounding problem을 분석한다.
- Taylor series expansion으로 task loss를 근사함으로써, rounding task를 quadratic (이차) unconstrained binary optimization problem으로 정의한다. 이를 layer-wise local loss로 단순화하고 soft relaxation을 활용하여 최적화하는 방법을 제안한다. 
- AdaRound는 단순한 반올림 방식보다 성능이 크게 향상될 뿐만 아니라, 여러 네트워크와 task에서 post-training quantization의 state-of-the-art를 달성한다. 
- Fine-tuning없이도 ResNet18 및 ResNet50의 가중치를 $4$-bit로 양자화하면서도 정확도 손실을 $1$ 이내로 유지했다.

$\textbf{Notation}$

- $x, y$: input and target variable 
- $\mathbb{E}$: expectiation operator
- $\mathbf{W}_{i,j}^{(l)}$: weight matrix (or tensor)
- $\mathbf{w}^{(l)}$: flattened version of $W^{(l)}$
- All vectors: column vectors and small bold letters $\mathbf{z}$ 
- All matrices (or tensors): capital bold letters $\mathbf{Z}$
- Functions $f()$
- Task loss $L$
- Constant $s$



***

### <strong>Motivation</strong>

- Rounding-to-nearest가 왜 optimal이 아닌지에 대한 직관적인 이해를 위해 pretrained model의 weight를 perturb시켰을 때 무슨 일이 발생하는지를 알아보자.
- Neural network가 flattened weights $\mathbf{w}$에 의해 parameterized됐다고 가정하자.
  - $\Delta \mathbf{w}$: a small perturbation
  - $L(x, y, \mathbf{w})$: task loss that we want to minimize
  - (a): second order Taylor series expansion
  - $g^{(\mathbf{w})}, H^{\mathbf{w}}$: expected gradient and Hessian of the task loss

***

### <strong>Method</strong>


***

### <strong>Experiment</strong>


***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>



![](img_path)
<a href="">link</a>


> 인용구
