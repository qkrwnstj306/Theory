<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
*ICCV(2017), 19262 citation*

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
<strong>"Visualize Attention Map via Gradients Equal to Importance of Channel like Weights"</strong></br>
</div>

***

### <strong>Intro</strong>


***

### <strong>Related Work</strong>


***

### <strong>Method</strong>

- CAM 과 달리, GAP (Global Average Pooling) 에 구애받지 않고 사용할 수 있다. 즉, 모델 구조가 제한적이지 않음 

<p align="center">
<img src='./img2.png'>
</p>

- 모델이 예측한 class 값을 feature map 값에 대해 편미분을 진행한다. 
  - 특정 channel pixel value 의 activation value 가 바뀜에 따라 class $y^c$ 의 값이 많이 흔들린다는 얘기는, class $y^c$ 를 결정할 때 해당 channel 의 영향력이 크다는 얘기이다. (즉, gradient 값의 크기가 크다) 
  - 이때, gradient 의 값의 부호와는 상관이 없다고 볼 수 있다. Gradient 가 음수인 큰 값이어도 이 값은 class 를 결정하는데 중요하다. (Negative Influence)
  - CAM 의 구조에선, CAM 에서의 weight 와 동일한 수식으로 볼 수 있다.

<p align="center">
<img src='./img3.png'>
</p>

- Class $y^c$ prediction 할 때의 feature (channel) map 에다가, 각 channel 의 영향력과 동일한 의미를 가진 gradient 값들을 곱해줌으로써 중요도를 반영한 class activation map 을 만들 수 있다. 
  - ReLU 를 취해서 음의 값은 무시한다. 

<p align="center">
<img src='./img4.png'>
</p>

***

### <strong>Experiment</strong>


<p align="center">
<img src='./img5.png'>
</p>

***

### <strong>Conclusion</strong>

- 한 가지 의문점은 CAM 에서의 weight 와 동일한 역할을 하는 gradient 가 모든 case 에 대해서는 적용되지 않는다는 점이다. 
  - 우리는 앞서, gradient 의 부호와는 상관없이 크기가 그 channel 의 영향력을 정하는 척도임을 알 수 있었다. 
    - 실제로 어느 방향이로든 update 를 진행하지 않기 때문에, 고정된 지점에서 바라봤을 때 크기로 영향력을 체크해야 한다. 
  - 하지만, channel 값이 해당 class 에 대해서 중요한 값이라고 가정했을 때
    1. gradient 는 음수인 큰 값, channel 은 양수인 값이라면 오히려 시각화가 되지 않는다.
    2. 반대로 gradient 는 양수인 큰 값, channel 은 음수인 값이라면 오히려 시각화가 되지 않는다. 
    3. 즉, channel 은 중요한데 시각화가 되지 않는다는 말이다. 
    - 이는 gradient 에 ReLU 를 취한 Grad-CAM++ 이나 임의로 activation fucntion 을 ReLU 로 써도 해결되지 않는 문제이다. 


| always gradient is positive | positive | negative | always channel is positive | positive | negative |
|:-------------------------:|:--------:|:--------:|:--------------------------:|:--------:|:--------:|
|      gradient * channel     | positive | negative |      gradient * channel      | positive | negative |

- 아는 지인들과 토론 $4$ 시간을 해도, 답이 안나와서 교수님께도 여쭤봤는데 역시나 동일
  - 카이스트가 구현한 코드에는 수식의 ReLU 대신, abs() 를 사용했는데 이러면 가능하다. 
  - 즉, abs() 를 사용하면 모든 case 에서 사용할 수 있다. 

***

### <strong>Question</strong>


