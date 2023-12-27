<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Theory%20Review&fontSize=50" width="100%">


## SDE: Stochastic Differential Equation
*확률 미분 방정식*

[Definition](#definition)</br>




***

### <strong>Definition</strong>

- White Gaussian Noise (백색 가우시안 잡음)
  - 통신 시스템에서, 열잡음이 신호에 영향을 주는 특성에 따라 붙여진 이름이다. 모든 주파수에 걸쳐서 나타난다.
    - 열 에너지에 의해 발생하는 것으로 저항기에서 많이 발생하며 기기 내부 잡음의 주요한 원인이 된다.
  - Stochastic process $\text{Noise} \ n(t) \in R^d, \ t \in [t_0, \infin ]$ 가 다음을 만족할 때 White Gaussian Noise 라고 부른다.
    - 1. $t_1 \neq t_2$ 일 때, $n(t_1), n(t_2)$ 가 독립
      - 즉, noise 가 더해지는게 시간 축과는 상관없다.
    - 2. $n(t)$ 가 Gaussian Process
    - 3. $E[n(t)] = 0, \ E[n(t)n(s)^T] = 0 \ (\text{if} \ t \neq s, \ E[n(t)n(t)^T] = Q)$
      - 여기서 $Q$ 는 spectral density 라고 부른다.
  - 특징
    - $n(t)$ 는 $t$ 에 관해서 불연속이다. 
    - $n(t)$ 는 $[t_0, \infin)$ 의 모든 부분 구간 $[t_1,t_2], \ t_0 \leq t_1 \leq t_2$ 안에서 unbounded 이고 아주 큰 값과 아주 작은 값을 갖는다.

- Brownian Motion (브라운 운동)
  - Stochastic process $w(t) \in R^d, \ t \in [t_0, \infin]$ 가 다음을 만족할 때, Brownian motion 이라고 부른다.
    - 1. $s < t$ 일 때, $w(t) - w(s) \sim N(0, (t-s)Q)$
    - 2. $w(t_0) = 0$
    - 3. $t_1 < t_2 < t_3 < \cdots < t_N$ 에 대하여 $w(t_2) - w(t_1), w(t_3) - w(t_2), \cdots , w(t_N) - w(t_{N-1})$ 는 독립이고 가우시안 분포를 따른다.
      - 여기서 $Q$ 는 diffusion matrix 라고 부른다.
  - $s < t$ 일 때, $w(t) - w(s) \sim N(0, (t-s)Q)$ 이고, 둘의 차이가 $1$ 이라면, $N(0,Q)$ 의 분포를 따른다. 
    - $N(0,Q)$ 의 분포는 $\text{if} \ t \neq s, \ E[n(t)n(t)^T] = Q$  와 같다.
  - 특징
    - $w(t)$ 는 $t$ 에 관해서 미분 불가능하다.
    - White Gaussian noise $n(t)$ 의 specatral density 가 $Q$ 라면 

$$ n(t) = \frac{dw(t)}{dt} \ [\textbf{Weak Derivative}]$$

- Ito Integration (이토 적분)
  - $G(t) \in R^d \times R^d$ 를 stochastic process 라고 가정.
  - $w(t) \in R^d$ 를 Brownian motion 이라고 가정.
  - $\int_S^T G(t) dw(t) = \lim_{n \rightarrow \infin} \displaystyle\sum_{i=0}^{n-1} G(t_i)(w(t_{i+1}) - w(t_i))$
    - $S = t_0 < t_1 < \cdots < t_n = T$
    - 여기서 $G(t)$ 에 $t=t_i$ 가 들어오는 것에 주목.
    - $t_i$ 는 partition $(t_i, t_{t+1})$ 에서 왼쪽 점이다.
    - $\displaystyle\sum_{i=0}^{n} G(t_i^*)(w(t_{i+1}) - w(t_i)), \ t_i \leq t_i^* \leq t_{i+1}$ 임의의 점이면 적분이 존재하지 않을 수 있다.







***

- Langevin Dynamics Sampling
  - 특정 데이터 확률 분포인 $p(x)$ 에서 sampling 하고 싶을 때 사용하는 방법이다.
  - $z_i$ 는 평균이 $0$ 이고 covariance 가 $I$ 인 가우시안 분포를 따른다.
  - $T > 0$ 가 엄청 크고, $\epsilon > 0$ 이 엄청 작으면 원하는 분포 $p(x)$ 로부터 데이터를 sampling 할 수 있다. 
  - $for \ i=1,2,3, \cdots T$

$$ x_i = x_{i-1} + \frac{\epsilon}{2} \nabla_x \log{p(x)} + \sqrt{\epsilon}z_i $$

$$ x_i - x_{i-1} = \frac{1}{2} \nabla_x \log{p(x)}\epsilon + \sqrt{\epsilon}z_i $$

$$ dx = \frac{1}{2} \nabla_x \log{p(x)}dt + dw $$

- 즉, $dt = \epsilon$, $dw = \sqrt{\epsilon}z_i \sim N(0,\epsilon)$ 를 만족하는 *SDE* 로 볼 수 있다.
