<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## LLaMA: Open and Efficient Foundation Language Models
*arXiv(2023), 8982 citation, Meta AI, Review Data: 2024.10.06*

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

$\textbf{이 주제의 문제점과 기존의 노력들}$

$\textbf{최근 노력들과 여전히 남아있는 문제들}$

$\textbf{본 논문에서 해결하고자 하는 문제와 어떻게 해결하는지, 그 결과들}$

$\textbf{본 논문의 주요 기여점}$

***

### <strong>Related Work</strong>


***

### <strong>Method</strong>

**Implementation detail**

<a href='https://www.youtube.com/watch?v=Mn_9W1nCFLo'>Youtube Link</a>

$\textsf{Architectural differences between the vanilla Transformer and LLaMA}$

- Difference: LLaMA 기준에서 설명
  - Block전에 RMS Noramlization을 수행한다.
  - (Rotary) Positional encoding은 Query & Key에 대해서만 한다.
  - Vanilla transformer는 feed forward에 ReLU를 사용하지만 LLaMA는 SwiGLU를 사용한다.

<p align="center">
<img src='./img2.png'>
</p>

- Hyper-parameters
  - Dimension: the size of the embedding vector
  - Heads: attention heads
  - Layers: the number of the block

<p align="center">
<img src='./img3.png'>
</p>

- Versus LLaMA 2

<p align="center">
<img src='./img4.png'>
</p>

$\textsf{Input Embedding}$


<p align="center">
<img src='./img5.png'>
</p>

$\textsf{RMS Normalization (with review of Layer Normalization)}$

- LayerNorm의 잘 알려진 성공의 이유는 re-centering & re-scaling invariance property이다. 입력 데이터보다 learnable parameter (scaling & shifting factor)에 의존
  - Re-centering invariance: input이 shift되어도 평균을 0으로 맞추기 때문에 학습에 영향을 주지 않는다.  
  - Re-scaling invariance: input이 특정 상수배로 크기가 변하더라도 LayerNorm은 이를 표준편차로 나눠서 정규화하므로, 학습에서 크기 변화에 민감하게 반응하지 않는다. 

- 하지만, <a href='https://arxiv.org/pdf/1910.07467'>RMS LayerNorm</a> 에서는 re-centering invariance보다 re-scaling invariance가 더 긍정적인 영향을 가지고 있다고 가정했다.
  - 따라서 re-scaling만을 진행

<p align="center">
<img src='./img6.png'>
</p>

$\textsf{Rotary Positional Embeddings}$

- Vanilla Transformer: Absolute positional encoding

<p align="center">
<img src='./img7.png'>
</p>

- Transformer의 위치 정보를 추가해주기 위해 사용되는 position representation을 절대적인 위치 정보가 아닌 상대적인 위치 정보로 나타낸다는 아이디어를 제시
  - <a href='https://arxiv.org/pdf/1803.02155'>Self-Attention with Relative Position Representations</a>
  - Self-attention에 각각의 input element끼리의 관계를 고려하는 거리 개념을 추가 

<p align="center">
<img src='./img8.png'>
</p>

<a href='https://arxiv.org/pdf/2104.09864'>RoFormer: Rotary Position Embedding</a> 

...continue

$\textsf{KV-Cache}$

$\textsf{Multi-Query Attention}$

$\textsf{Grouped Multi-Query Attention}$

$\textsf{SwiGLU Activation Function}$


***

### <strong>Experiment</strong>


***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>



![](img_path)
<a href="">link</a>


> 인용구
