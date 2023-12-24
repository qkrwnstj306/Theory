<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Theory%20Review&fontSize=50" width="100%">


## Pixel-Wise Independent Learning
*in image generative models*

[Definition](#definition)</br>




***

### <strong>Definition</strong>

- 생성 모델에서 구하고자 하는 것은 데이터의 분포인 $p(x)$ 이다. 이 $p(x)$ 에서 확률 변수는 이미지 $x$ 이다. 이때, $p(x)$ 는 각 픽셀들을 확률 변수로 보고 다변수 확률 분포 (확률 변수가 여러개)로 표현할 수 있다. 또한, 다변수 확률 분포를 표현하기 위해 결합확률분포 (확률 변수들이 동시에 발생) 로 볼 수 있다. 이때, 일반적으로 각 픽셀간의 독립을 가정한다. 즉, 각 픽셀이 생성될 때 서로 영향을 주지 않음.

$$ p(x) = p(x_1,x_2, \cdots , x_{100}) = p(x_1) \times p(x_2) \times \cdots \times  p(x_{100})  \\ where, \  x \in R^{10 \times 10} $$

- 이미지의 차원이 $10 \times 10$ 이라면, 이미지 $x$ 는 $100$ 차원의 한 점이다. 우리는 이 점들의 분포를 알고 싶은 것이다. (e.g., 이미지 $1000$ 장의 sample($1000$ 개의 점) 을 가지고 있다면 이 $1000$ 장은 $p(x)$ 에서 sampling 된 값으로 볼 수 있다.) 이때, 이 점은 $10 \times 10$ 이라는 high dimension 을 가지는 점이다.
- 그럼 이미지 생성 모델에서, 결합확률분포에 대해 학습을 진행하는가? 
  - 아니다. pixel-wise independent learning (이후에 설명) 으로 인해, 사실상 $p(x_1,x_2, \cdots, x_{100})$ 을 한 번에 학습하는 게 아니라 $p(x_1), p(x_2),\cdots, p(x_{100})$ 을 각각 학습한다.
- 그럼 이미지에 노이즈를 추가할 때, 각 픽셀에 대해 서로 다른 노이즈를 더하는 이유는 뭘까? (같은 분포에서 sampling 된 노이즈) 
  - 바로, 이미지가 스칼라가 아닌 픽셀들의 배열이기 때문이다. 대부분의 경우, 각 픽셀은 해당 위치에 대한 정보를 나타내고 이미지를 생성하는 동안 각 픽셀은 모델에 의해 독립적으로 조작되고 생성되며 (각 픽셀은 pixel-wise independent learning 을 가정. 확률에서의 독립과는 다르다.) 다른 픽셀에 대한 정보나 특성은 직접적으로 공유되거나 전파되지 않는다. 이러한 독립적인 처리 방식은 모델이 이미지의 다양한 부분을 동시에 고려하고, 각 픽셀에 대해 독립적으로 다양한 특성이 생성될 수 있도록 한다. 또한, 이미지 생성에 있어서 모델이 더욱 복잡하고 다양한 패턴을 학습할 수 있다.
  - $Noise \sim N(0,I)$ 표준 정규 분포일 때, 노이즈가 더해진 이미지는 여전히 다변량 가우시안 분포이자 결합 확률 분포로 표현이 가능하다. 이때의 각 픽셀의 평균은 이미지를 sampling 했을 때의 픽셀 값 즉, $x_1,x_2,x_3, \cdots$ 가 된다. 분산은 동일하고 각 픽셀이 확률적으로 독립이니 완전한 원의 분포 형태를 가진다.

<p align="center">
<img src='./img1.png'>
</p>

<p align="center">
<img src='./img2.png'>
</p>

- pixel-wise independent learning 은 각 픽셀에 대한 확률 분포를 독립적으로 학습하는 것이다. 즉, 각 픽셀에 대한 확률 변수를 각각 독립적으로 모델링하고 학습한다. 따라서 $p(x_1,x_2,x_3, \cdots)$ 를 한 번에 학습하는 것이 아닌, $p(x_1),p(x_2),p(x_3), \cdots$ 처럼 개별적으로 학습을 한다. 이렇게 하면 각 픽셀의 특징과 패턴을 독립적으로 학습할 수 있고, 모델 구현이 단순해진다. 반면에 이 방식은 픽셀 간의 상호 작용을 무시하게 되므로, 이미지의 공간적 구조나 의존성을 고려하지 않는다는 단점이 존재한다. 따라서 문제에 맞춰 픽셀 간의 관계를 잘 모델링 할 수 있는 방법이나 구조를 고려해야 한다.
  - 예로, 모델 구조를 통해 pixel-wise independent learning 의 단점을 개선할 수 있다. Kernel 을 이용해 인접한 pixel 간의 공간적 정보를 담을 수 있는 CNN 이나 attention 구조를 통해 다른 pixel 간의 관계를 살펴볼 수도 있다.
- 그럼 왜 픽셀마다 개별적인 학습인가? 
  - loss fucntion 을 뜯어 보면 자세히 알 수 있다. 일반적으로 픽셀마다 값을 뽑은 뒤에 (이미지니까 $2$ or $3$ 차원) 같은 차원의 어떤 값 (e.g., noise) 를 빼주고 전체적으로 평균을 낸다. 이 과정에서 특정 픽셀이 다른 픽셀들과 상호작용을 한 것은 평균을 내기 위한 덧셈과 나눗셈이다. 실제로 결합확률로 해당 픽셀값을 샘플링하려면 그에 맞는 수식을 바탕으로 풀어써야하는데 그런게 없이 단일 확률에서 샘플링했다.
  - 개별적인 학습이 이해가 안된다면 generative model 의 목적 함수에서 일반적으로 나타나는 $\Vert \text{objective fucntion} \Vert_2^2$ 을 보자. Norm 을 구하는 것 자체가 이미 각 픽셀을 학습과정에서 개별적인 원소로 보고 각 확률적인 오차를 제곱하여 계산한다는 의미이다. 제곱하고 더하는 이 연산이 과연 픽셀 간의 상호작용을 고려한 것일까.

> Objective Function of Denoising Score Matching 

<p align="center">
<img src='./img3.png'>
</p>

> Objective Function of DDPM

<p align="center">
<img src='./img4.png'>
</p>