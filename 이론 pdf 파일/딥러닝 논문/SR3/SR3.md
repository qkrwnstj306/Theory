![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


## SR3: Image Super-Resolution via Iterative Refinement
*IEEE Transactions on Pattern Analysis and Machine Intelligence(2022), 666 citation*

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
<strong>"Super-Resolution via DDPM"</strong></br>
</div>

***

### <strong>Intro</strong>
- DDPM(stochastic iterative denoising process) 을 이용하여 image-to-image translation (super-resolution) 을 한다. 
  - U-Net 구조 및 pure Gaussian noise, iterative refinement
  - Low-resolution image(resize 해서 high dim 으로 맞춰줌) & high-resolution image 가 concat 되어 input 으로 들어간다. 
- Face 뿐만 아니라, natural image 도 잘 작동한다.
- $8 \times$ face super-resolution task on CelebA-HQ: fool rate $50\%$ (GAN baselines: fool rate $34\%$)
- $4 \times$ super-resolution task on ImageNet: baseline 과 비교했을 때, human evaluation 과 high-resolution image 로 학습된 ResNet-50 classifier 의 classification accuracy 를 능가했다. 
- $256 \times 256$ ImageNet image generation challenge 에서도 generative model 을 super-resolution model 에 연결시켜, 경쟁력 있는 FID score 를 달성.  Cascaded image generation 에서도 효율성을 입증했다. 

***

### <strong>Related Work</strong>


***

### <strong>Method</strong>

<p align="center">
<img src='./img2.png'>
</p>

***

### <strong>Experiment</strong>


***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>


