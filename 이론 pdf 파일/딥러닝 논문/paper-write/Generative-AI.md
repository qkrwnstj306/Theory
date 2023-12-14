![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


## Generative AI
*Junseo Park*

[Intro](#intro)</br>
[Related Work](#related-work)</br>
[Method](#method)</br>
[Experiment](#experiment)</br>
[Conclusion](#conclusion)</br>

> Core Idea
<div align=center>
<strong>"Classification Operations in the Generative AI Model (생성 모델의 분류 작업)"</strong></br>
</div>

***
### <strong>CVPR & ICCV Trend<strong>
*2023 keyword*
- Diffusion
- 3D
- Human pose, face, body, movements, gesture, etc.
- Image & Video generation
- NeRF
- Multimodal

*2023 comment*
- Image style transfer 분야는 흥미를 잃어간다.
  - 특히 image style transfer using GAN 은 잘 안한다.
- Video/3D style transfer 는 가능

***
### <strong>Image Synthesis<strong>
- <a href='../DDPM_231112_174348.pdf'>DDPM</a>
- <a href='../DDIM/DDIM.md'>DDIM</a>
- <a href='../Latent Diffusion Model_230315_164557.pdf'>LDM</a>
- LCM
- <a href='../Generative Adversarial Nets_230512_221610.pdf'>GAN</a>
- <a href='../VAE_230706_175059.pdf'>VAE</a>
- DALL-E2
- Imagen
- CogView2
- Parti
- Muse

### <strong>Controlling Image Diffusion Models/Personalization</strong>

<a href='https://arxiv.org/pdf/2307.06949.pdf'>reference paper-HyperDreamBooth</a>

<a href='https://arxiv.org/pdf/2306.00983.pdf'>reference paper-StyleDrop</a>

- <a href='/이론 pdf 파일/딥러닝 논문/DreamBooth/DreamBooth.md'>DreamBooth</a>
  - Optimize entire T2I network weights
  - 관심 주제를 설명하는 몇 장의 이미지에 대해 entire text-to-image model 을 fine-tuning 한다.
  - 따라서 표현력이 풍부하며 세부적인 내용으로 주제를 캡쳐할 수 있다.
  - Language drift & overfitting 을 완화하기 위해 class-specific prior preservation loss & class-specific prior image 를 도입했다.
- <a href='/이론 pdf 파일/딥러닝 논문/Textual_inversion/Textual_inversion.md'>Textual Inversion</a>
  - Optimize text embedding
- <a href='/이론 pdf 파일/딥러닝 논문/LoRA_231007_193848.pdf'>LoRA</a>
  - Optimize low-rank approximations of weight residuals
- Custom Diffusion
  - Optimize cross-attention layers
- SVDiff
  - Optimize singular values of weights
- <a href='/이론 pdf 파일/딥러닝 논문/StyleDrop_230626_130819.pdf'>StyleDrop</a>
  - Use adapter tuning and finetunes a small set of adapter weights for style personalization
  - Generation in any style with a single input image
  - Style 특화 personalization. 다른 방법들은 painting style 에 제한되어 있지만, StyleDrop 은 a single style reference image 만을 사용하여 3D rendering, design illustration, sculpture 등 다양한 시각적 스타일을 시연한다.
- DreamArtist
  - One-shot personalization techniques by employing a positive-negative prompt tuning strategy
- HyperNetwork
  - Use an auxiliary neural network to predict network weights in order to change the functioning of a specific neural network
- <a href='/이론 pdf 파일/딥러닝 논문/ControlNet_231018_163832.pdf'>ControlNet</a>
  - Propose ways to incorporate new input conditioning such as depth
- Specialist Diffusion

- StyleBoost
  - Style personalization with T2I diffusion models using DreamBooth
  - *vs StyleDrop & DreamBooth*


***

### <strong>Style Transfer</strong>
#### Traditional Method (e.g., CNN, GAN, ViT)
- 확인 필요
- StyleCLIP
- VQGAN-CLIP
- CLIPStyler
- StyleGAN
- Style-NaDa

- 확인 완료
- AdaIN
- Neural Style Transfer

#### Diffusion-based model
- <a href='/이론 pdf 파일/딥러닝 논문/PatchMatch/PatchMatch.md'>Diffusion-Enhanced PatchMatch</a>
  - Utilize Patch-based techniques with whitening and coloring transformations in latent space  
- <a href='/이론 pdf 파일/딥러닝 논문/StyleDiffusion/StyleDiffusion.md'>StyleDiffusion</a>
  - 해석가능하고 제어가능한 C-S disentanglement and style transfer 를 제안했다.
  - CLIP image space 에서 C-S 를 해결하기 위해 CLIP-based style disentanglement loss 와 style reconstruction prior 를 도입했다. 
- <a href='/이론 pdf 파일/딥러닝 논문/Inversion-based-style-transfer/Inversion-based-style-transfer.md'>Inversion-based Style Transfer</a>
  - textual description $[C]$ 학습
  - Inference 시에, $[C]$ 와 content image 의 inversion 을 통해 생성
- <a href='../DreamStyler_231001_142049.pdf'>DreamStyler</a>
  - Advanced Texutal Inversion
  - Using BLIP-2, image encoder
  - Text 에 style binding

***

### <strong>Super Resolution</strong>
#### Diffusion-based model
- <a href='../SR3/SR3.md'>SR3</a>


***

### <strong>Image-to-Image Translation</strong>
- Pix2Pix
- Cycle-GAN
- Plug-and-Play

### <strong>Image Editing</strong>
- <a href='../Multimodal Garment Designer_231015_171057.pdf'>Multimodal Garment Designer</a>
- <a href='../Prompt to prompt image editing with attention_231004_124613.pdf'>Prompt-to-Prompt</a>
- <a href='../InstructPix2Pix_231014_141433.pdf'>InstructPix2Pix</a>

### <strong>Data Augmentation</strong>


***

### <strong>Inpainting</strong>

#### Diffusion-based model
- <a href='../Repaint/RePaint  Inpainting using Denoising Diffusion Probabilistic Models.md'>RePaint</a>

***

### <strong>Question</strong>


