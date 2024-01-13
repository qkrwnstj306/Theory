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
  - VITON-HD
  - Dress Code
  - Single Stage Virtual Try-on via Deformable Attention Flows
  - High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions
  - C-VTON: Context-Driven Image-Based Virtual Try-On Network
  - TryOnDiffusion
  - GP-VTON
  - Stable VITON
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

$\textbf{We Want}$

1. 데이터 증강 및 condition 정보를 잘 받아들여서 (강건한 학습), $(1)$ 적은 데이터셋이나 $(2)$ 배경이 복잡, $(3)$ condition 에 옷만 있는 게 아닌 사람도 존재, $(4)$ input 과 condition 이 같은 사람이여도 처리가능한 모델
2. 상하의 다 적용가능할 뿐만 아니라, full body 여도 처리가능한 모델

$\textbf{Solution}$

- Method.            
  - 1. Controlnet.             
  - 2. $+$ condition.            
    - ControlNet 은 Image-based VITON 에 적합한 구조는 아니다.
    - Input 에 간단한 network 추가
  - 3. $+$ model architecture
    - 입력을 줄 때, noise 뿐만 아니라 다른 정보들도 concat 을 하는데 이때 이 정보들을 짧게 처리해서 넣어주는 건 정보를 온전히 받아들일 수 없다. 또한, Stable-VITON 의 경우, 앞의 network 는 freeze 라 정보 손실이 존재한다. 즉, Condition의 정보를 제대로 흡수해야한다
  - 4. 옷 사진만이 아니라 옷을 입고 있는 사람을 condition으로도 받고 싶음. 즉, 복잡한 배경의 condition. 심지어 input 과 condition 이 같은 사람이여도 처리
    - Augmentation.(crop/rotate/upsample) for robustness
      - crop 하고 upsampling 
      - reference image 에 사람이 있어도 crop 하고 upsampling
      - reference image rotation 등의 augmentation 
      - <a href='../../딥러닝 논문/Paint-by-Example/Paint-by-Example.md'>Painting by example 참고</a>

- Additional method
  - 1. Loss function 
  - 2. Image encoder

$\textbf{Therefore}$

1. Robust (augmentation, 모델 구조 개선을 통한 확실한 정보 흡수) 
2. Reasonable (method 전개) 
3. Generalization (상하의 가능, 복잡한 배경, 풀바디)