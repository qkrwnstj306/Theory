<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## [paper]
*NeurIPS(2023), 96 citation, UC Berkeley & Google Research, Review Data: 2024.06.20*

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
<strong>"Guide Sampling using the Attention and Activations of a Pretrained Diffusion Model, Self-Guidance"</strong></br>
</div>

***

### <strong>Intro</strong>

$\textbf{이 주제의 정의 및 요구사항과 중요한 이유}$

- Large-scale generative model은 상세한 text description으로부터 high-quality 이미지를 생성할 수 있다. 


$\textbf{이 주제의 문제점과 기존의 노력들}$

- 하지만, 이미지의 여러 측면을 텍스트를 통해 전달하는 것은 어렵거나 불가능하다. 
- 이 문제를 해결하기위해, 
  1. Detail을 더 잘 조절하기 위해 pretrained model을 조정했다.
     1. Textual Inversion 
     2. Imagic
     3. DreamBooth
  2. Textual prompt와 함께 reference image를 공급했다. 
     1. Text2live
     2. Instructpix2pix
  3. 다른 종류의 conditioning을 공급했다.
     1. Palette
     2. ControlNet

- 그럼에도, 이러한 접근법들은 광범위한 paired data (따라서 가능한 수정 영역이 제한된다)으로 학습하거나, 몇 가지 image manipulation을 수행하기 위해 비싼 optimization process를 수행해야 했다. 
- 몇몇 방법들은 output을 설명하는 target caption을 사용하여 zero-shot으로 input image를 수정할 수 있었지만, 오직 limited control만 가능하다. 
  - Structure-preserving appearance manipulation이 제한되거나 uncontrolled image-to-image translation이다. 


$\textbf{최근 노력들과 여전히 남아있는 문제들}$

- 결과적으로, 많은 단순한 편집들이 여전히 손이 닿지 않는 곳에 남아있다. 
  - 어떠한 것도 바꾸지 않고 한 object를 움직이거나 크기를 조절할 수 있을까
  - 한 이미지의 객체 외관을 다른 이미지에 복사할 수 있을까
  - 하나의 장면 레이아웃과 다른 이미지의 외관을 결합할 수 있을까
  - 특정 아이템이 캔버스의 특정 위치에 정확한 모양으로 배치된 이미지를 생성할 수 있을까

$\textbf{본 논문에서 해결하고자 하는 문제와 어떻게 해결하는지, 그 결과들}$

- Internal representation of diffusion model을 guiding하여 생성된 이미지 전반에 걸쳐 조작하는 방법인 self-guidance를 도입한다. 
  - Shape, location, and appearance of object 와 같은 특징이 이러한 representation 으로부터 추출될 수 있고, sampling process를 조종하는데 사용할 수 있다. 
  - Self-guidance는 classifier guidance와 비슷하게 작동하지만, 추가적인 model이나 학습이 없이 pretrained model 자체에 제공된 signal을 사용한다. 

- 본 논문은 간단한 속성 집합을 구성하는 것만으로도 특정 물체의 size나 position을 수정하는 것, 한 이미지 안에 서로 다른 이미지의 layout + apperance of objects 를 결합하는 것, multiple image 로부터 물체들을 한 이미지 안에 구성하는 것 등의 image manipulation을 수행한다.
- 또한, self-guidance가 real image를 수정하는 데 사용될 수 있음을 보여준다. 


***

### <strong>Related Work</strong>


***

### <strong>Method</strong>

$\textbf{Guidance}$

- 앞의 두 항은 classifier-free guidance를 나타낸다.
  - 마지막 항의 $g(z_t,;t, y)$는 energy function

<p align="center">
<img src='./img2.png'>
</p>


***

### <strong>Experiment</strong>


***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>



![](img_path)
<a href="">link</a>


> 인용구
