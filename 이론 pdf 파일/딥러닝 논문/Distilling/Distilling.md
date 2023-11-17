![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


## Distilling the Knowledge in a neural newtork
*arXiv(2015), 16596 citation*

[Intro](#intro)</br>
[Related Work](#related-work)</br>
[Method](#method)</br>
[Experiment](#experiment)</br>
[Conclusion](#conclusion)</br>

<p align='center'>
<img src='./img1.png'>
</p>

> Core Idea
<div align=center>
<strong>"Effectively Copy Teacher Model using Soft Target"</strong></br>
</div>

***

### <strong>Intro</strong>
- 현실에선, 복잡한 딥러닝 모델을 경량화된 디바이스에서 사용하기 위해, 낮은 메모리를 필요로 하면서 높은 정확도가 나오는 모델이 필요하다.
- 복잡한 딥러닝 모델(teacher model)은 많은 수의 파라미터를 가지고 있는 반면, 좋은 *Knowledge* 를 가지고 있다.
- 본 논문에서는, teacher model 의 knowledge 를 효과적으로 student model 에게 전달해주는 방식을 제공하고자 한다.

***

### <strong>Related Work</strong>


***

### <strong>Method</strong>
- Soft target 는 softmax function 에 temperature 를 도입해 확률 값이 극단적으로 나오는 걸 막아줄 수 있다.
- 이때, 확률 값이 smooth 가 되면서 정보량이 많아지는 효과가 발생하는데 이 많은 정보량을 바탕으로 student model 에게 전달해준다.

<p align='center'>
<img src='./img2.png'>
</p>
<p align='center'>
<img src='./img3.png'>
</p>
<p align='center'>
<img src='./img4.png'>
</p>
<p align='center'>
<img src='./img5.png'>
</p>
<p align='center'>
<img src='./img6.png'>
</p>

***

### <strong>Experiment</strong>
- MNIST dataset 에 대해서 학습을 할 때 (for classification), 숫자 $3$ image 를 학습 과정에서 사용하지 않았음에도 불구하고, test 시에 $3$ 을 $98.6%$ 로 맞췄다.
<p align='center'>
<img src='./img7.png'>
</p>

***

### <strong>Conclusion</strong>
- 거대 모델의 지식을 단순하면서도 효과적으로 전이할 수 있는 방법이다.

***

### <strong>Question</strong>



