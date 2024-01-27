<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Theory%20Review&fontSize=50" width="100%">


## VITON (Virtual Try On)
*가상 옷 fitting*

<a href='https://github.com/minar09/awesome-virtual-try-on'>Reference Github</a>

[Definition](#definition)</br>




***

### <strong>Definition</strong>
- Target person 에게 원하는 옷을 가상으로 입히는 task 이다.

### <strong>Type of Task</strong>
- Prompt-based Virtual Try On
  - <a href='../../딥러닝 논문/ControlNet_231018_163832.pdf'>ControlNet</a>
  - <a href='../../딥러닝 논문/Multimodal Garment Designer_231015_171057.pdf'>Multimodal Garment Designer</a>
  - EditAnything
- Image-based (2D) Virtual Try On
  - VITON
  - CP VITON
  - LA-VITON
  - CP-VTON+
  - <a href='../../딥러닝 논문/VITON-HD/VITON-HD.md'>VITON-HD</a>
  - Dress Code
  - Single Stage Virtual Try-on via Deformable Attention Flows
  - High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions
  - C-VTON: Context-Driven Image-Based Virtual Try-On Network
  - <a href='../../딥러닝 논문/TryOnDiffusion/TryOnDiffusion.md'>TryOnDiffusion</a>
  - GP-VTON
  - DCI-VTON
  - LADI-VTON
  - <a href='../../딥러닝 논문/stableVITON/stableVITON.md'>Stable VITON</a>
- Multi-Pose Guided Virtual Try On
- Video Virtual Try On
- Non-clothing Virtual Try On
- Pose-Guided Human Synthesis
- Datasets for Virtual Try On
- In the Wild Virtual Try On
- Etc. (3D Virtual Try On, Mix-and-match Virtual Try On ...)

### <strong>Image-based (2D) Virtual Try On</strong>

#### <strong>Difficulty</strong>
- 옷을 입혔을 때의 artifacts
- 체형이 바뀜에 따른 옷의 변화 (주름, 근육의 표현 등)
- 옷의 길이에 따른 변화
  - 옷의 길이 조절
  - Target person 이 긴팔을 입고 있는데, 반팔을 입히려면 target person 이 반팔을 부족한 정보 (e.g., 팔의 모양, 길이, 포즈, etc.)를 model 이 채워야한다. 
- 문신 및 시계 등 악세서리의 보존
  - 생성 시에 없어지는 경우가 많다. 
- target person 과 그 배경은 유지한채 옷만 바꿔야 한다.

#### <strong>Training Approach</strong>

#### <strong>Dataset</strong>

1. VITON
2. Dress Code


#### <strong>Evaluation</strong>

#### <strong>My Method</strong>

$\textbf{Problem 및 전개}$

1. 기존의 방법들: $2$ stage system
   1. 우리가 ai를 쓰는 이유는 복잡한 전처리(hand crafted feature) 를 하지 않고, 간단하면서도 간편한 입력만으로도 처리할 수 있는 능력을 가진 ai를 활용하는 것. 하지만 기존의 방법들은 투스테이지로, 후처리도 해야하고, 학습할때 모델 여러개를 복합적으로 엮어서 해야한다.

2. Diffusion Model 의 등장
   1. 최근에는 diffusion model이 강력하면서도 놀라운 생성능력을 보여줌.

3. Controlling Generative model.
   1. Controlnet 및 다양한 통제 기법들 소개하면서 controlnet 장단점 자세히 설명
   2. 하지만 controlnet도 모델 구조상 Image-based VITON task에 적합하진 않다.
      1. Text driven 이기 때문에 input 이 image 가 아닌 noise 이다. Noise 에서 시작함
      2. 심지어 text 는 해당 task 에 필요가 없다.
      3. Input 에 noise 하나만 들어가서 (적어도, cloth-agnoistic image 가 같이 들어가야 된다) target person 의 정보를 줄 수 없으니 target person 자체를 정의할 수도 없고 보존할 수도 없다. 

$\textbf{We Want}$

1. 데이터 증강 및 condition 정보를 잘 받아들여서 (강건한 학습), $(1)$ 적은 데이터셋이나 $(2)$ 배경이 복잡, $(3)$ condition 에 옷만 있는 게 아닌 사람도 존재, $(4)$ input 과 condition 이 같은 사람이여도 처리가능한 모델
2. 상하의 다 적용가능할 뿐만 아니라, full body 여도 처리가능한 모델

$\textbf{Solution}$

- Method.            
  - 1. Controlnet.
    - ControlNet 은 입력 제어 맵에서 내용을 인식하는 능력이 강력하다. 즉 강력한 인식 능력을 가지고 있어서 어려운 조건 (e.g., image 가 불명확한 것처럼 정보가 부족한 상황)에서도 잘 작동한다. 
    - 사실 다른 인코더도 사용할 수 있지만, 목표에 따라 다르다. 즉, 제한 조건이 많은 해당 task 의 경우 강력한 인코더가 필요하다.             
      - <a href='https://github.com/lllyasviel/ControlNet/discussions/188'>관련 실험</a>
  - 2. $+$ condition.            
    - ControlNet 은 Image-based VITON 에 적합한 구조는 아니다.
    - Input 에 간단한 network 추가 (concat 을 처리하는 용도)
  - 3. $+$ model architecture
    - 입력을 줄 때, noise 뿐만 아니라 다른 정보들도 concat 을 하는데 이때 이 정보들을 짧게 처리해서 넣어주는 건 정보를 온전히 받아들일 수 없다.
    - 예시로, Stable-VITON 의 경우 initial conv 로 concat 한 정보들을 처리해서 SD Encoder 로 넣어주려면 $4$ channel 로 압축해야하는데 이는 정보 손실이 일어나는 구간 (병목현상)으로 볼 수 있고, 게다가 SD Encoder network 는 freeze 라 concat 한 새로운 정보를 제대로 받지 못할 수 있다. 즉, Condition의 정보를 제대로 흡수해야한다
  - Training Approach: 
    - Fine-tuning vs Parameter-efficient 
      - Use LoRA
    - Freeze U-Net vs Unfreeze
      - Out-of-Distribution data 가 들어오면 U-Net 은 생성 능력이 떨어진다. 따라서 U-Net 도 학습은 해줘야 한다. 
    - Train Cross-attention/Self-attention
      - Existing SD model 은 text-encoder 에 대해서 학습되었다. 즉, 적어도 cross attention 은 학습해야 하고자하는 task 에 맞춰서 domain shift 가 될 것이다.
        - Ref. IP-Adapter: text encoder $\rightarrow$ image encoder 로 바꿈으로써 cross-attention 학습  
      - Self-attention: 마찬가지로 학습 data 가 fashion image 이므로 fashion domain 에 맞게 학습해야 한다. 
      - Ref. prompt-to-prompt: spatial layout and geometric 이 cross-attention 에 의존한다. 
      - Ref. Textual Inversion: text embedding 만 학습하기 때문에, text embeding 을 아무리 최적화 시켜도 diffusion model 이 학습한 distribution 에 없으면 생성하기가 힘들다. 따라서 diffusion model 도 어떤 방식으로든 학습시켜야한다. 
  - 4. 옷 사진만이 아니라 옷을 입고 있는 사람을 condition으로도 받고 싶음. 즉, 복잡한 배경의 condition. 심지어 input 과 condition 이 같은 사람이여도 처리
    - Augmentation.(crop/rotate/upsample) for robustness
      - crop 하고 upsampling 
      - reference image 에 사람이 있어도 crop 하고 upsampling
      - reference image rotation 등의 augmentation 
      - <a href='../../딥러닝 논문/Paint-by-Example/Paint-by-Example.md'>Painting by example 참고</a>

- Additional method
  - 1. Loss function 
  - 2. Image encoder

- Tip
  - One-stage 로 하려면 결국 warped cloth image information 을 implicit 하게 학습해야 한다. 그러기 위해선 사람 이미지 정보를 줘야 한다. 
  - E.g., StableVITON: ControlNet input 에 사람 정보 더해주기

$\textbf{Final Method}$

- Input: Noise, agnoistic map, mask, dense pose
- Controlnet input: 의류 사진 (augmentation)
- Image encoder?
- 아니면, CLIP image encoder 만 사용?

$\textbf{First Experiment}$

1. Stable diffusion model v1.5 + CLIP image encoder + Zero-kernel + classifier-free guidance

2. + LR_scheduler, Augmentation, target person info

3. vs full fine-tuning

**고려 사항**
- Uncond probability: $0.1$ / $0.2$


$\textbf{Therefore}$

1. Robust (augmentation, 모델 구조 개선을 통한 확실한 정보 흡수) 
2. Reasonable (method 전개) 
3. Generalization (상하의 가능, 복잡한 배경, 풀바디)
4. Parameter efficient fine-tuning