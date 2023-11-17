![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


## SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
*PMLR(2020), 13261 citation*

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
<strong>"Learn Visual Representation using Contrastive Learning(Cosine-Similarity) 

[self-supervised learning]"</strong></br>
</div>

***

### <strong>Intro</strong>
- 한 장의 사진을 data augmentation 을 통해, 다른 view 를 만들어내고 해당 pair image 는 positive 라고 정의한다. 또한, 마찬가지로 다른 data 들도 같은 방식으로 pair 쌍이 존재할텐데 그 data 들은 negative data 라고 정의한다(처음 한 장의 사진에 대해서). 
<p align="center">
<img src='./img2.png'>
</p>

***

### <strong>Related Work</strong>


***

### <strong>Method</strong>
- Encoder output 이 image representation 이라고 가정하고 downstream task 에 사용한다. 
- 마지막 단의 Dense layer 의 activation fuctnion 이 없는 이유는 cosine-similarity 는 음수 값이 나올 수 있기 때문이다. 

<p align="center">
<img src='./img3.png'>
</p>

- Data augmentation 의 경우, Color distort(e) & Crop(c) 를 같이 썼을 때 성능이 가장 좋았다.
<p align="center">
<img src='./img4.png'>
</p>
<p align="center">
<img src='./img5.png'>
</p>

- Batch size 가 2라면 다음과 같은 matrix 가 나오고 column & row 별로 cosine-similarity 를 계산한다. 
<p align="center">
<img src='./img6.png'>
</p>
<p align="center">
<img src='./img9.png'>
</p>
<p align="center">
<img src='./img10.png'>
</p>
<p align="center">
<img src='./img8.png'>
</p>
***

### <strong>Experiment</strong>
- Batch size 가 크면 클수록 성능이 일반적으로 올라간다.
<p align="center">
<img src='./img11.png'>
</p>

***

### <strong>Conclusion</strong>
- 유사도를 측정하는 cosine-similarity 를 사용했지만, 특정 이미지와 얼마나 다른 지는 측정할 수 없다. 
  - 밀어내는 정도를 학습하거나 지정할 수 없기 때문에..!
- Batch size 를 크게 해서 negative sample 들이 많은 상태가 좋다. 

***

### <strong>Question</strong>

