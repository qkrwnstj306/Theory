<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Theory%20Review&fontSize=50" width="100%">


## SDE: Stochastic Differential Equation
*확률 미분 방정식*

[Definition](#definition)</br>




***

### <strong>Definition</strong>

- Langevin Dynamics Sampling
  - 특정 데이터 확률 분포인 $p(x)$ 에서 sampling 하고 싶을 때 사용하는 방법이다.
  - $z_i$ 는 평균이 $0$ 이고 covariance 가 $I$ 인 가우시안 분포를 따른다.
  - $T > 0$ 가 엄청 크고, $\epsilon > 0$ 이 엄청 작으면 원하는 분포 $p(x)$ 로부터 데이터를 sampling 할 수 있다. 
  - $for \ i=1,2,3, \cdots T$

$$ x_i = x_{i-1} + \frac{\epsilon}{2} \nabla_x \log{p(x)} + \sqrt{\epsilon}z_i $$

$$ x_i - x_{i-1} = \frac{1}{2} \nabla_x \log{p(x)}\epsilon + \sqrt{\epsilon}z_i $$

$$ dx = \frac{1}{2} \nabla_x \log{p(x)}dt + dw $$

- 즉, $dt = \epsilon$, $dw = \sqrt{\epsilon}z_i \sim N(0,\epsilon)$ 를 만족하는 *SDE* 로 볼 수 있다.
