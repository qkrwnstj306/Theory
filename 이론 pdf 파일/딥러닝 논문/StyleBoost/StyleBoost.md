![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


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
<strong>"test1"</strong></br>
</div>

***

### <strong>Intro</strong>


***
### Test

- Text-to-Image Models
  - Recent Advancements in text-to-image models, exemplified by models like Imagen, DALL-E2, Stable Diffusion (SD), Muse, and Parti, demonstrate impressive image generation capabilities when provided with a textual prompt.

- Personalization/Controlling Generative Models
  - *The exploration of personalized text-to-image synthesis has been a subject of research, aiming to enhance and control images of personal assets through the utilization of pre-trained text-to-image models. Textual inversion serves as a key technique, facilitating the discovery of text representations (e.g., embedding, token) corresponding to a set of images without necessitating changes to the parameters of the text-to-image model.*
  - DreamBooth, a representative model in this category, fine-tunes the entire text-to-image model using a small set of images that describe the subject of interest. This approach enhances expressiveness and enables detailed subject capture. To address language drift and overfitting, DreamBooth introduces class-specific prior preservation loss and class-specific prior image. Further advancements in parameter-efficient fine-tuning (PEFT), such as LoRA or adapter tuning, contribute to efficiency improvements.
  - Additionally, methods like Custom Diffusion and SVDiff extend the capabilities of DreamBooth, enabling the simultaneous synthesis of multiple subjects. In contrast, the innovative StyleDrop model diverges by building upon Muse, a generative vision transformer, rather than relying on text-to-image diffusion models. StyleDrop utilizes adapter tuning and fine-tunes a small set of adapter weights, facilitating style personalization. Notably, it allows the generation of content in various visual styles using a single style reference image, spanning domains like 3D rendering, design illustration, and sculpture.
  - Several other models, including ControlNet, HyperNetwork, DreamArtist, and Specialist Diffusion, propose diverse techniques such as incorporating new input conditioning, predicting network weights through auxiliary neural networks, implementing one-shot personalization strategies, optimizing cross-attention layers, and optimizing singular values of weights.

- Style Transfer(Diffusion-based)
  - *While both StyleDrop and traditional Neural Style Transfer (NST) produce stylized images, they differ fundamentally. StyleDrop, based on text-to-image models, generates content, whereas NST relies on an image to guide content synthesis based on spatial structure.*
  - Under the category of style transfer, traditional methods (CNN/GAN based) and diffusion-based methods, including Diffusion-Enhanced PatchMatch, StyleDiffusion, and Inversion-based Style Transfer, present diverse techniques. 
  - <CNN/GAN based description>
  - Diffusion-Enhanced PatchMatch employs patch-based techniques with whitening and coloring transformations in latent space. StyleDiffusion proposes interpretable and controllable content-style disentanglement, addressing challenges in CLIP image space. Meanwhile, Inversion-based Style Transfer focuses on utilizing textual descriptions for synthesis.
  - During the inference stage, models like DreamStyler showcase advanced textual inversion, leveraging techniques such as BLIP-2 and an image encoder to generate content through the inversion of text and content images while binding style to text.
  - *In our pursuit of artistic style-specific personalization, we conduct experiments with human subjects, distinguishing our method from others. Notably, our approach incorporates actual human images in the training process, setting it apart in the realm of personalization studies.*
 

***

### <strong>Related Work</strong>
- Text-to-Image Models 
  - Several recent models such as Imagen, DALL-E2, Stable Diffusion(SD), Muse, Parti etc. demonstrate excellent image generation capabilities given a text prompt. 
- Personalizaiton/Controlling Text-to-Imamge Models 
  - Personalized Text-to-Image Synthesis has been studied to edit images of personal assets by leveraging the power of pre-trained text-to-image models. Textual inversion made easy find text representations (e.g., embedding, token) corresponding to a set of images of an object without changing parameters of the text-to-image model.
  - DreamBooth fine-tunes an entire text-to-image model on a few images describing the subject of interest. As such, it is more expressive and captures the subject with greater details. 
  - Parameter-efficient fine-tuning (PEFT) methods, such as LoRA or adapter tuning, are adopted to improve its efficiency.
  - Custom diffusion and SVDiff have extended DreamBooth to synthesize multiple subjects simultaneously. Unlike these methods built on text-to-image diffusion models, we build StyleDrop on Muse, a generative vision transformer.

- Style Transfer
  -  While both output stylized images, StyleDrop is different from NST in many ways; ours is based on text-to-image models to generate content, whereas NST uses an image to guide content (e.g., spatial structure) for synthesis. 

- Personalization/Controlling Text-to-Image Models
    1. DreamBooth
        - Optimize entire T2I network weights
        - 관심 주제를 설명하는 몇 장의 이미지에 대해 entire text-to-image model 을 fine-tuning 한다.
        - 따라서 표현력이 풍부하며 세부적인 내용으로 주제를 캡쳐할 수 있다.
        - Language drift & overfitting 을 완화하기 위해 class-specific prior preservation loss & class-specific prior image 를 도입했다.
    2. Textual Inversion
        - Optimize text embedding
        - 적은 양의 이미지로 text-to-image model 의 parameter 변화 없이 text representation 을 찾는다. (e.g., token embedding)
    3. LoRA
        - Optimize low-rank approximations of weights residuals
        - Parameter efficient fine-tuning (PEFT) methods
    4. StyleDrop
        - Use adapter tuning and finetunes a small set of adapter weights for style personalization
        - Generation in any style with a single input image
        - Style 특화 personalization. 다른 방법들은 painting style 에 제한되어 있지만, StyleDrop 은 a single style reference image 만을 사용하여 3D rendering, design illustration, sculpture 등 다양한 시각적 스타일을 시연한다.
    5. ControlNet
        - Propose ways to incorporate new input conditioning such as depth
    6. HyperNetwork
        - Use an auxiliary neural network to predict network weights in order to change the functioning of a specific neural network
    7. DreamArtist
        - One-shot personalization techniques by employing a positive-negative prompt tuning strategy
    8. Custom Diffusion
        - Optimize cross-attention layers
    9.  SVDiff
        - Optimize singular values of weights
    10. Specialist Diffusion   


- Style Transfer
    1. Traditional Method (CNN/GAN based)
    2. Diffusion-based Method
       1. Diffusion-Enhanced PatchMatch
          - Utilize Patch-based techniques with whitening and coloring transformations in latent space
       2. StyleDiffusion
          - 해석가능하고 제어가능한 C-S disentanglement and style transfer 를 제안했다.
          - CLIP image space 에서 C-S 를 해결하기 위해 CLIP-based style disentanglement loss 와 style reconstruction prior 를 도입했다.
       3. Inversion-based Style Transfer
          - textual description $[C]$ 학습
          - Inference 시에, $[C]$ 와 content image 의 inversion 을 통해 생성
       4. DreamStyler
          - Advanced Texutal Inversion
          - Using BLIP-2, image encoder
          - Text 에 style binding

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

