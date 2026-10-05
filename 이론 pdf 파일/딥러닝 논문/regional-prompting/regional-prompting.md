<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">


## Training-free Regional Prompting for Diffusion Transformers
*arXiv(2024), 30 citation, Peking, UC Berkeley, CMU, Review Data: 2026.07.26*

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

- Diffusion model의 text-to-image generation 능력은 완벽한 능력을 보여주고 있다. Semantic understanding (i.e., prompt following) 능력 또한 large language models (e.g., T5, Llama)와 함께 향상되고 있다. 

- 하지만, 기존의 모델들은 와벽하게 길고 복잡한 text prompt를 다루지 못한다. 특히 text prompt가 수 많은 attributes와 상호연결된 공간적인 관계의 다양한 objects를 포함하면 그렇다. 

- 많은 regional prompting method들은 UNet-based models (SD1.5, SDXL)에 대해 제안됐지만, 최근 DiT 구조 (SD3나 FLUX.1)를 기반으로는 구현되지 않았다. 

- 따라서 본 논문에서는 trainin-free로 fined-grained compositional T2I generation을 가능하게 하는 FLUX.1 based attention manipulation **regional prompting** 을 제안 및 구현한다.  

<p align="center">
<img src='./img2.png'>
</p>


***

### <strong>Related Work</strong>


***

### <strong>Method</strong>

<p align="center">
<img src='./img3.png'>
</p>

- 본 연구의 주요 목적은 별도의 학습 없이 텍스트 조건과 공간 조건을 모두 반영하도록 조합적 생성 능력을 향상시키는 것이다. 

- Input 
    - global description
    - $N$ tuples: $(c_i, m_i)$ - sub region의 description 및 binary mask 
    - 



***

### <strong>Experiment</strong>


***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>



![](img_path)
<a href="">link</a>


> 인용구
