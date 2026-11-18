<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Theory%20Review&fontSize=50" width="100%">


## Probability
*조건부 확률, 결합 확률*

[Definition](#definition)</br>




***

### <strong>Definition</strong>

- 나만의 정의
  - 사건이 일어났을 때 (발생했을 때) 특정 사건 집합에서 내가 고려중인 사건이 발생할 확률
  - 특정 사건 집합: 일반적으로 전체/ 조건부 확률일 때는 조건 집합
  - 내가 고려중인: 확률 변수

- 결합 확률: 사건이 일어났을 때, 전체 사건 집합에서 내가 고려중인 확률 변수 (e.g., $x$ , $y$) 가 모두 발생할 확률

- 조건부 확률: 사건이 일어났을 때, 조건 집합 중 (e.g., $x$) 에서 내가 고려중인 확률 변수 (e.g., $y$) 가 발생할 확률
  - $x$ 중에서만 봐! 라고 가이드를 해주는 꼴이다.
  - **조건부 확률은 시간축과 인과관계를 고려하지 않는 확률이다. 우리가 흔히 착각하는 게 $x$ 가 먼저 일어나고 $y$ 가 일어날 것이라고 가정하는데 이는 틀린 가정이다.**
    - <a href='https://m.blog.naver.com/drgon09/221939826287'>2022 개정 수학과 교육과정 관련 내용</a>
    - E.g., 항아리에 흰 공 $2$ 개와 검은 공 $2$ 개가 있다고 하자. 공을 다시 집어넣지 않고 공을 하나씩 두 번 꺼낸다. 두 번째로 꺼낸 공이 흰색이었다고 할 때, 첫 번째로 꺼낸 공이 흰색일 확률은 얼마인가
    - 답은 $\frac{1}{2}$ 이 아닌, $\frac{1}{3}$ 이다. 
    - 즉, 나중에 일어난 사건이라도 먼저 일어난 사건에 영향을 줄 수 있다는 반례이다.

- 결합 확률 분포
  - 이산 확률 변수일 경우, $p_{XY}(x,y)$ 로 표기
  - 연속 확률 변수일 경우, $f_{XY}(x,y)$ 로 표기
  - 여기서의 확률 변수는 $x$ 와 $y$ 이다.

$$ p_{XY}(x,y) = P[X = x, Y = y] $$

$$ 0 \leq p_{XY}(x,y) \leq 1 \ for \ all \ x \ and \ y $$

$$ \Sigma_x \Sigma_y p_{XY}(x,y) = 1 $$

$$ 0 \leq f_{XY}(x,y)  $$

$$ \int_{-\infin}^{\infty} \int_{-\infin}^{\infty} f_{XY}(x,y) dx dy = 1 $$


$$ P(a \leq X \leq b, c \leq Y \leq d) =  \Sigma_{a \leq x \leq b} \Sigma_{c \leq y \leq d} p_{XY}(x,y) $$

$$ P(a \leq X \leq b, c \leq Y \leq d) =  \int_{a}^{b} \int_{c}^{d} f_{XY}(x,y) dx dy  $$


- 기댓값

$$ E[g(X,Y)] = \Sigma_x \Sigma_y g(x,y)p_{XY}(x,y) \\ = 

\int_{-\infin}^{\infty} \int_{-\infin}^{\infty} g(x,y)f_{XY}(x,y) dx dy $$