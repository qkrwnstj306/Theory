<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold
*SIGGRAPH conference(2023), 94 citation, Max Planck Institute for Informatics (in Germany), Review Data: 2024.05.10*

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
<strong>"test1"</strong></br>
</div>

***

### <strong>Intro</strong>

$\textbf{이 주제의 정의 및 요구사항과 중요한 이유}$

- User 의 요구를 만족시키는 시각적 content 를 합성하는 것은 pose, shape, expression, and generated object 의 layout 을 정확하고 유연하게 control 해야한다. 
  - Social-media user 는 사람/동물의 position, shape, expression, body pose 를 조절하기를 원한다. 
  - 자동차 디자이너는 자신의 작품의 형태를 쌍방향으로 수정하고 싶다. 
  - 이러한 user 의 요구사항을 만족시키기 위해, 이상적인 조절 가능한 이미지 합성 접근법은 다음의 특성을 갖춰야한다.
    1. Flexibility: model 은 생성된 물체의 layout, pose, position, expression, shape 등을 포함한 서로 다른 공간적 특성을 조절할 수 있어야 한다. 
    2. Precision: 높은 정확도로 공간적 특성을 조절해야 한다. 
    3. Generality: 특정 카테고리에만 제한되지 않고 다양한 카테고리에도 적용할 수 있어야한다. 

$\textbf{이 주제의 문제점과 기존의 노력들}$

- 기존의 접근법은 GAN 과 주석이 달린 training data 를 사용하는데, 오직 하나 또는 $2$ 개의 요구사항을 만족한다. (Flexibility, Precision, Generality)
- Prior $3$ D model 을 통한 GAN 의 조작이나 주석이 달린 train data 에 의존하는 supervised learning: 새로운 object category 에 대해서 일반화가 부족하여 종종 Flexibility 가 부족하다. 

$\textbf{최근 노력들과 여전히 남아있는 문제들}$

- 최근에는, text-guided image synthesis 가 주목을 받았지만 text guidance 는 precision 과 flexibility 가 부족하다. 
  - E.g., 특정 pixel 로의 움직임을 조절할 수 없다. 

$\textbf{본 논문에서 해결하고자 하는 문제와 어떻게 해결하는지, 그 결과들}$

- 따라서, Flexibility, Precision, Generality 을 모두 만족하는 DragGAN 을 제안한다. 
  - Handle point 를 "drag" 하여 target point 에 정확하게 도달하게 한다. 

- 이 접근법은 마치 UserControllableLT 와 비슷하다. 이것도 마찬가지로 dragging-based manipulation 이지만 이것과 비교하여, 본 논문은 더 어려운 문제를 해결하고자 한다.
  1. One point 뿐만 아니라 더 많은 point 조절 (이전에는 못했음)
  2. Target point 에 정확하게 도달하는 handle point (이전에는 못했음)

- 본 논문에서 해결할 문제는 다음과 같다.
  1. Handle point 를 target point 로 이동시키는 feature-based motion supervision
     1. Latent code 를 최적화시키는 shifted feature patch loss 제안
     2. 흥미가 있는 영역만 수정하는, region-specific editing 제안
  2. Generator feature 를 이용하여 handle point 의 위치를 추적하는 point tracking approach

- 다양한 dataset (animals: lions, dogs, cats, and horses / humans: face and whole body) 에 대해서 광범위한 평가를 진행했다. 
  - User-defined handle point 를 target point 로 효과적으로 옮긴다. 
  - 많은 obejct category 에 대해서 다양한 조작 효과를 달성한다. 

- GAN 의 학습된 generative image manifold (의미를 보존하는 공간) 에서 이러한 조작들이 수행된다.  
  - Network 의 추가학습이나 주석이 달린 data 가 필요없다. 

***

### <strong>Related Work</strong>


***

### <strong>Method</strong>


***

### <strong>Experiment</strong>


- 더 많은 handle point 를 다룰 수 있음을 보여준다.
  - 비슷한 이전 연구와의 차이점을 부각

<p align="center">
<img src='./img3.png'>
</p>

- Point tracking 이 baseline 과 비교하여 정확하게 일치함을 보여준다. 
  - 비슷한 이전 연구와의 차이점을 부각

<p align="center">
<img src='./img2.png'>
</p>

***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>

