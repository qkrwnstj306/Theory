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
<strong>"Guidnace without Additional Classifier"</strong></br>
</div>

***

### <strong>Intro</strong>


***

### <strong>Related Work</strong>


***

### <strong>Method</strong>

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
