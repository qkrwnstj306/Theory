![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


## RePaint: Inpainting using Denoising Diffusion Probabilistic Models
*CVPR(2022), 441 citation*

[Intro](#intro)</br>
[Related Work](#related-work)</br>
[Method](#method)</br>
[Experiment](#experiment)</br>
[Conclusion](#conclusion)</br>

![result](./img1.png)

> Core Idea
<div align=center>
<strong>"Image Inpainting using DDPM"<strong></br>
without Condition such as Text
</div>

***

### <strong>Intro</strong>
<p> 
- Image inpainting 은 masking 된 부분을 생성했을 때, 나머지 영역과 의미적으로 합리적이어야 한다. 심지어는 극단적인 형태의 masked region 도 처리해야 한다.
- 대부분의 존재하는 방법들은 특정 분포의 마스크를 학습하기 때문에, unseen mask type 에 대해서 일반화 능력이 부족하다.
- 본 논문에서는, pretrained unconditional DDPM 을 사용하여 free-form inpainting task 를 해결하고자 한다. Probabilistic modeling 에 기반함에도 (not deterministic) diverse and high-quality image를 생성할 수 있다.
- Inpainting task 를 학습하지 않는다. 이로 인해 얻을 수 있는 이점은 2가지 이다.
	- inference 시에 어떤 형태의 mask 가 들어와도 일반화 할 수 있다.
	- image 생성에 강력한 능력을 갖춘 DDPM 을 사용했기 때문에, 더욱 의미론적인 이미지 생성을 할 수 있다. 
- Standard DDPM sampling strategy 를 사용하면 texture 는 일치하지만, 종종 의미적으로 부정확할 때가 존재한다. 따라서 본 논문에서는 개선된 denoising 전략(= **resample**)을 도입한다.
</p>

***

### <strong>Related Work</strong>
<p>
test1</br>
</p>

***

### <strong>Method</strong>
<p>
test1</br>
</p>

***

### <strong>Experiment</strong>
<p>
test1</br>
</p>

***

### <strong>Conclusion</strong>
<p>
- Iterative 하게 Random Gaussian noise 를 sampling 했기 때문에 다양한 output image 를 생성할 수 있다. 
- free-form inpainting 이 가능하다. (masking 모양에 대한 generalization 능력을 갖춤)
</p>

***

### <strong>Question</strong>
<p>
test1</br>
</p>

![](img_path)
<a href="">link</a>


> 인용구
