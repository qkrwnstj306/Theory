<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## [paper]
*CVPR(2022), 441 citation*

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
<strong>"Trainable Copy and Zero Convolution"</strong></br>
</div>

***

### <strong>Method</strong>

- Zero Conv 가 SD encoder Block 마다 있다는 걸 유념해야 한다. 
  - E.g., SD Encoder Block A 가 $3$ 개가 있다면 zero conv 도 $3$ 개가 있다. 
  - Concat 이나 cross-attetion 이 아닌 add 이다. 

