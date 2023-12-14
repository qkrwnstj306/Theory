<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50" width="100%">

## StyleBoost: ?
*?(?), ? citation*

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
<strong>"훈련 과정에서 사람 이미지를 적극적으로 도입/DreamBooth 의 class-prior image 변형/Style personalization"</strong></br>
</div>

***

### <strong>Intro</strong>


***

### <strong>Related Work</strong>
- **DreamBooth & StyleDrop 에 대한 detail description -> 비교를 위해**
- **Style Transfer 와의 다른 점 부각**

- **Text-to-Image Models**
  - Models like Imagen, DALL-E2, Stable Diffusion (SD), Muse, and Parti lead recent advancements in the field of text-to-image synthesis. Particularly, these models demonstrate remarkable image generation capabilities when presented with textual prompts. The textual prompts serve as guiding mechanisms, allowing these models to transform written descriptions into visually appealing images. This functionality represents a significant advancement, empowering users to influence and enhance the generated visual content using the expressive power of natural language.
  - [Imagen, DALL-E2, Stable Diffusion(SD), Muse 및 Parti와 같은 모델은 텍스트-이미지 합성 분야에서 최근 발전을 주도하고 있습니다. 특히 이러한 모델은 텍스트 프롬프트가 표시될 때 놀라운 이미지 생성 기능을 보여줍니다. 텍스트 프롬프트는 안내 메커니즘 역할을 하여 이러한 모델이 서면 설명을 시각적으로 매력적인 이미지로 변환할 수 있도록 해줍니다. 이 기능은 사용자가 자연어의 표현력을 사용하여 생성된 시각적 콘텐츠에 영향을 미치고 향상시킬 수 있다는 점에서 상당한 발전을 의미합니다.]

- **Personalization/Controlling Generative Models**
  - With the development of the field of text-image synthesis, methods of personalization to suit the user's needs have emerged one after another.
  - [텍스트-이미지 합성 분야의 발전에 따라, 사용자의 요구에 맞게 개인화를 시키는 방법들도 연달아 등장했습니다.]
  - The exploration of personalized text-to-image synthesis has been a subject of research, aiming to enhance and control images of personal assets through the utilization of pre-trained text-to-image models. Textual inversion is used as a key technique to facilitate the discovery of text representations (e.g. embeddings for special tokens) corresponding to a set of images.
  - [개인 맞춤형 텍스트-이미지 합성의 탐구는 연구 주제로 자리잡고 있으며, 사전 훈련된 텍스트-이미지 모델을 활용하여 개인 자산의 이미지를 향상하고 제어하려는 목표를 가지고 있습니다. 여기서 텍스트 역전은 핵심 기술로 사용되며, 이는 특정 이미지 세트에 해당하는 텍스트 표현을 발견하기 위한 것입니다. (예: 특수 토큰에 대한 임베딩)]
  - DreamBooth, a representative model in this category, fine-tunes the entire text-to-image model using a small set of images that describe the subject of interest. This approach enhances expressiveness and enables detailed subject capture. To address language drift and overfitting, DreamBooth introduces class-specific prior preservation loss and class-specific prior image. Further advancements in parameter-efficient fine-tuning (PEFT), such as LoRA or adapter tuning, contribute to efficiency improvements.
  - [이 범주에서 대표적인 모델인 DreamBooth는 관심 대상을 설명하는 소량의 이미지를 사용하여 전체 텍스트-이미지 모델을 미세 조정합니다. 이 방식은 표현력을 향상시키고 세부 주제를 자세히 포착할 수 있게 합니다. 언어의 변화와 오버피팅을 해결하기 위해 DreamBooth는 클래스별 사전 보존 손실과 클래스별 사전 이미지를 도입합니다. 또한 LoRA나 어댑터 튜닝과 같은 매개변수 효율적 미세 조정(PEFT)의 추가 발전은 효율성 향상에 기여합니다.]
  - Additionally, methods like Custom Diffusion and SVDiff extend the capabilities of DreamBooth, enabling the simultaneous synthesis of multiple subjects. In contrast, the innovative StyleDrop model diverges by building upon Muse, a generative vision transformer, rather than relying on text-to-image diffusion models. StyleDrop utilizes adapter tuning and fine-tunes a small set of adapter weights, facilitating style personalization. Notably, it allows the generation of content in various visual styles using a single style reference image, spanning domains like 3D rendering, design illustration, and sculpture.
  - [또한 Custom Diffusion 및 SVDiff와 같은 방법은 DreamBooth의 기능을 확장하여 여러 주제를 동시에 합성할 수 있게 해줍니다. 이와 대조적으로 혁신적인 StyleDrop 모델은 텍스트-이미지 확산 모델에 의존하는 대신 생성적 비전 변환기인 Muse를 기반으로 분기됩니다. StyleDrop은 어댑터 튜닝을 활용하고 작은 어댑터 무게 세트를 미세 조정하여 스타일 개인화를 촉진합니다. 특히, 단일 스타일 참조 이미지를 사용하여 3D 렌더링, 디자인 일러스트레이션, 조각 등 다양한 영역에 걸쳐 다양한 시각적 스타일의 콘텐츠를 생성할 수 있습니다.]
  - Several other models, including ControlNet, HyperNetwork, DreamArtist, and Specialist Diffusion, propose diverse techniques such as incorporating new input conditioning, predicting network weights through auxiliary neural networks, implementing one-shot personalization strategies, optimizing cross-attention layers, and optimizing singular values of weights.
  - [ControlNet, HyperNetwork, DreamArtist, 그리고 Specialist Diffusion을 포함한 여러 모델은 새로운 입력 조건 통합, 부가 신경망을 통한 네트워크 가중치 예측, 일회성 개인화 전략 도입, 교차 주의 계층 최적화, 그리고 가중치의 특이값 최적화와 같은 다양한 기술을 제안합니다.]

- **Style Transfer**
  - While both **[Ours]** and traditional Neural Style Transfer (NST) produce stylized images, they differ fundamentally. **[StyleBoost]**, based on text-to-image models, generates content, whereas NST relies on an image to guide content synthesis based on spatial structure.
  - [**[Ours]**와 전통적인 Neural Style Transfer (NST)은 모두 스타일화된 이미지를 생성하나, 근본적으로 다릅니다. **[StyleBoost]**는 텍스트-이미지 모델을 기반으로 내용을 생성하며, NST는 이미지를 사용하여 공간 구조를 기반으로 내용 합성을 안내합니다.]
  - Under the category of style transfer, traditional methods (CNN/GAN based) and diffusion-based methods, including Diffusion-Enhanced PatchMatch, StyleDiffusion, and Inversion-based Style Transfer, present diverse techniques. 
  - [스타일 전송 범주에서 전통적인 방법(CNN/GAN 기반)과 확산 기반 방법, 그 중 Diffusion-Enhanced PatchMatch, StyleDiffusion, 그리고 Inversion-based Style Transfer를 포함한 다양한 기술들이 제시되고 있습니다.]
  - **<CNN/GAN based description>**
  - Diffusion-Enhanced PatchMatch employs patch-based techniques with whitening and coloring transformations in latent space. StyleDiffusion proposes interpretable and controllable content-style disentanglement, addressing challenges in CLIP image space. Meanwhile, Inversion-based Style Transfer focuses on utilizing textual descriptions for synthesis.
  - [Diffusion-Enhanced PatchMatch는 잠재 공간에서 화이트닝 및 컬러링 변환을 사용하는 패치 기반 기술을 도입합니다. StyleDiffusion은 CLIP 이미지 공간에서의 도전 과제를 해결하기 위해 해석 가능하고 제어 가능한 콘텐츠-스타일 이탈을 제안합니다. 한편, Inversion-based Style Transfer는 합성을 위해 텍스트 설명을 활용하는 데 중점을 둡니다.]
  - During the inference stage, models like DreamStyler showcase advanced textual inversion, leveraging techniques such as BLIP-2 and an image encoder to generate content through the inversion of text and content images while binding style to text.
  - [추론 단계에서 DreamStyler와 같은 모델들은 BLIP-2와 이미지 인코더와 같은 기술을 활용하여 텍스트 및 콘텐츠 이미지의 역전을 통해 콘텐츠를 생성하고 스타일을 텍스트에 바인딩하는 고급 텍스트 역전을 시연합니다.]


- In our pursuit of artistic style-specific personalization, we conduct experiments with human subjects, distinguishing our method from other personalization methods. Notably, our approach actively incorporates human images into the training process, setting it apart uniquely in the realm of personalization studies.
- [우리는 예술적 스타일에 특화된 개인화를 추구하면서 인간을 대상으로 실험을 진행하며 우리의 방식을 다른 개인화 방식과 구별합니다. 특히, 우리의 접근 방식은 인간의 이미지를 훈련 과정에 적극적으로 통합하여 개인화 연구 영역에서 이를 독특하게 차별화합니다.]
- *여기서 StyleDrop & DreamBooth 를 언급하는게 자연스러운가?*

- List: [specialist,DreamArtist,textaul inversion,DreamBooth,StyleDrop,CustomDiffusion,SVDiff]
- (1): 사람 propmt 대해서 style 이 잘 입혀지는지 inference 를 했는가?
- (2): 사람 이미지를 학습 과정에서 사용했는가? for object
- (3): 사람 이미지를 학습 과정에서 사용했는가? for style

- 사람을 학습
  - 사람을 object 로 보고 object를 학습하는 것과
  - 사람 이미지를 artistic style 의 일부에 포함시켜, style 학습에 사람 이미지를 넣는 것은 다르다.
    - 사람 이미지를 style 의 일부로 포함시키는 것을 중요시 여기는 reference paper 는 찾지 못했다. 
    - 그럼, 우리는 왜 중요한 요소로 보고, 포함시켰는가?
      - 사람을 inference 단계에서 뽑아내는 건 흔한 일이다.
      - 사람을 학습시키는 것은 매우 어려운 일이다. 즉, 복잡한 관계를 학습해야 한다. 그러므로 course <-> fine-grained feature 를 다양하게 학습할 수 있다.
      - 사람을 포함해서 학습하면 FID 측면에서도 성능이 올라간다. 

- 근본적인 의문: 왜 사람 이미지를 style 의 일부로 포함시켜서 학습하지 않는가?
    - 우리가 실험해본 결과, 유의미한 결과를 가져온다. 
    - 즉, style personalization 에선 style 의 일부에 사람 이미지를 포함해야 한다!
    - Style personalization 을 주제로 잡은 StyleDrop 과 비교해야한다.

|     | Specialist Diffusion | DreamArtist | Textual Inversion | DreamBooth | StyleDrop | CustomDiffusion | SVDiff |
|-----|:--------------------:|:-----------:|:-----------------:|------------|:---------:|-----------------|--------|
| (1) |           O          |      O      |         X         |      X     |     O     |        X        |    X   |
| (2) |           X          |      O      |         O         |      X     |     X     |        X        |    X   |
| (3) |           X          |      X      |         O         |      X     |     O     |        X        |    X   |



***

**MEMO**

- 우리가 제시하고자 하는 것은 artistic style 특화 personalization 이다. 
    - 그 중에서도 사람에 대해서 실험
    - Our method 는 DreamBooth & StyleDrop 과 비슷 
    - DreamBooth architecture 를 사용
    - StyleDrop 처럼 style personalization
    - 하지만 사람에 대해서 집중한 점과 artistic style 에 대해서 집중한 점들이 다른 점이다. 실제 학습과정에서도 사람 이미지를 학습에 사용한다. 

> Personalization
>> StyleDrop & Textual Inversion & DreamBooth 는 스타일을 사람에 대해서 실험하지 않았다.

> Style Transfer
>> Diffusion-enhanced PatchMatch & DreamStyler 는 스타일을 사람에 대해서 실험하지 않았다.

***

### <strong>Method</strong>
- 넓은 범위의 artistic style 을 binding 시키겠다.
- artistic style 은 추상적이다. 
- DreamBooth 의 방법에서 class image/auxiliary image 는 학습 과정에서 영향을 끼치므로 중요하게 다뤄야 한다. 
  - Auxiliary image 를 freezed diffusion model 에서 생성하는 것이 아닌 고품질의 이미지로 대체한다. 
  - 이때, 이미지들은 target style 의 정보를 담고 있으면서도 더 일반적인 이미지로 구성한다. 
-  


***

### <strong>Experiment</strong>


***

### <strong>Conclusion</strong>


***

### <strong>Question</strong>

