<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Theory%20Review&fontSize=50" width="100%">


## Energy-based Model
*Normalizing Constant*

[Definition](#definition)</br>
[MLE](#)</br>
[MAP](#)</br>




***

### <strong>Definition</strong>

- Gibbs distribution
  
$$ p_{\theta}(x) = \frac{\exp{(-f_{\theta}(x))}}{Z_{\theta}} $$

- Normalizing Constant/Partition function

$$ Z_{\theta} = \int \exp(-f_{\theta}(x)) dx $$

- Maximum likelihood training: $\theta$ 에 대해서 미분

$$ \nabla_{\theta} \log{p_{\theta}(x)} = - \nabla_{\theta} f_{\theta}(x) - \nabla_{\theta} \log{Z_{\theta}} $$

- 이때, ${Z_{\theta}}$ 는 $\int p_{\theta}(x) dx = 1$ 을 만드는 **normalizing constant** 이다. $\theta$ 에 의존적인 이 값을 우리는 추정하기가 어렵다.

$$ \nabla_{\theta} \log{Z_{\theta}} = \nabla_{\theta} \log{\int \exp{(- f_{\theta}(x))dx}} $$

- $log$ 미분

$$ (\int \exp{(- f_{\theta}(x))dx})^{-1} \ \nabla_{\theta} \int \exp(- f_{\theta}(x))dx $$

- $\nabla_{\theta}$ 를 $\int$ 안에 넣어준다. ($\Sigma$ 로 생각하면 편함)

$$ (\int \exp{(- f_{\theta}(x))dx})^{-1} \  \int \nabla_{\theta} \exp(- f_{\theta}(x))dx $$

- $\exp$ 미분

$$ (\int \exp{(- f_{\theta}(x))dx})^{-1} \  \int \exp(- f_{\theta}(x)) (- \nabla_{\theta} f_{\theta}(x))dx $$

- $(\int \exp{(- f_{\theta}(x))dx}) = Z_{\theta}$ 이므로 $x$ 에 종속적이지 않다. 즉, $x$ 에 대해서 상수 취급을 할 수 있으므로 우측 텀의 $\int$ 안에 넣어준다. 


$$ \int (\int \exp{(- f_{\theta}(x))dx})^{-1} \ \exp(- f_{\theta}(x)) (- \nabla_{\theta} f_{\theta}(x))dx $$

- $(\int \exp{(- f_{\theta}(x))dx}) = Z_{\theta}$ 로 바꿔준다.

$$ \int \frac{1}{Z_{\theta}} \ \exp(- f_{\theta}(x)) (- \nabla_{\theta} f_{\theta}(x))dx $$

- $\frac{1}{Z_{\theta}} \ \exp(- f_{\theta}(x)) = p_{\theta}$ 이므로 치환

$$ \int p_{\theta}(x) (- \nabla_{\theta} f_{\theta}(x))dx $$

- 적분을 평균으로 변환

$$ E_{x \sim p_{\theta}(x)}[- \nabla_{\theta}f_{\theta}(x)] $$

- Monte Carlo 를 해서 sampling 을 통해 estimation 하면 될 것 같지만, EBM (Energy-based Model) 을 학습시키려면 EBM 에서 sampling 을 해야 한다. EBM 도 Diffusion-based model 처럼 iterative 하게 sampling 을 해야 하는데, 이걸 매 iteration 마다 sampling 해줘야 하니 엄청 느리다. 


> Normalizing constant 
>> 정규화 상수의 개념은 확률 이론 및 기타 영역에서 다양하게 발생한다. 정규화 상수는 확률 함수를 전체 확률이 $1$인 확률 밀도 함수로 변환하는 데 사용된다. 