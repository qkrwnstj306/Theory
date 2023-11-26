![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


## Classifier-Free Diffusion Guidance
*arXiv(2022), 937 citation*

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
<strong>"Guidance without Additional Classifier"</strong></br>
</div>

***

### <strong>Intro</strong>


***

### <strong>Related Work</strong>


***

### <strong>Method</strong>
- 다음의 식을 이용한다. <a href='../../딥러닝 이론/Score-based-generative-model/Score-based-generative-model.md'>reference</a>

$$ \nabla_x \log{p_{\sigma_i}(x)} \approx s_\theta(x_t,t\ or \ \sigma) = - \frac{1}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t\ or \ \sigma) $$

- Condition $y$ 를 given 으로 $y$ 를 반영해서 이미지를 생성하고 싶다면 다음과 같이 식이 전개된다.($\sigma$ 생략)

$$ \nabla_x \log{p(x_t|y)} = \nabla_x \log{\frac{p(x_t)p(y|x_t)}{p(y)}}  $$ 

$$ = \nabla_x \log{p(x_t)} + \nabla_x \log{p(y|x_t)} - \nabla_x \log{p(y)} $$

- $x$ 에 관한 함수가 아닌, $\nabla_x \log{p(y)}$ 는 소거된다.

$$ \nabla_x \log{p(x_t|y)} = \nabla_x \log{p(x_t)} + \nabla_x \log{p(y|x_t)} $$

- $\nabla_x \log{p(x_t)}$ 을 좌항으로 넘기면, 다음과 같이 만들어진다.
  
$$ \nabla_x \log{p(x_t|y)} - \nabla_x \log{p(x_t)}  = \nabla_x \log{p(y|x_t)} $$

- 위의 수식을 Classifier guidance 의 수식인 $\gamma\nabla_x \log{p(y|x_t)}$ 에 대입하게 되면,
  - 이때 대입하는  $\nabla_x \log{p(x_t|y)} - \nabla_x \log{p(x_t)}$ 는 근사값이다?

$$ \nabla_x \log{p(x_t|y)} = \nabla_x \log{p(x_t)} + \gamma\nabla_x \log{p(y|x_t)} $$

$$ \nabla_x \log{p(x_t|y)} = \nabla_x \log{p(x_t)} + \gamma(\nabla_x \log{p(x_t|y)} - \nabla_x \log{p(x_t)})$$

$$ =  \gamma\nabla_x \log{p(x_t|y)} + (1-\gamma)\nabla_x \log{p(x_t)} $$

- ????

$$ \tilde{\epsilon_\theta}(x_t, c) = (1+\gamma)\epsilon_\theta(x_t, c) - \gamma\epsilon_\theta(x_t) $$

- $w$ 는 조절 가능한 가중치이다. 논문에서는 $w=0.1$ or $0.3$ 일때 best FID result 를 얻었다고 한다. 
- IS 의 경우, $w >= 4$, 즉 trade-off 관계이다. 

<p align="center">
<img src='./img2.png'>
</p>

<p align="center">
<img src='./img3.png'>
</p>

<p align="center">
<img src='./img5.png'>
</p>

<p align="center">
<img src='./img4.png'>
</p>

- 결론적으로, 하나의 모델을 사용하되 unconditional model 의 경우 null token 을 조건 $c$ 로 주게 된다. 

<p align="center">
<img src='./img9.png'>
</p>

- 학습을 할때의 algorithm. DDPM 에서, condtion $c$ 를 넣는 것과 $p_uncond$ 의 확률로 condition $c$ 를 null 로만 바꿔주는 code 를 추가해주면 된다. 
<p align="center">
<img src='./img10.png'>
</p>


***

### <strong>Experiment</strong>

<p align="center">
<img src='./img6.png'>
</p>

<p align="center">
<img src='./img7.png'>
</p>

<p align="center">
<img src='./img8.png'>
</p>

***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>
