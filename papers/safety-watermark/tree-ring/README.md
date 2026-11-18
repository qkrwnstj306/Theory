# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust

*NeurIPS(2023), 391 citation, University of Maryland, Review Data: 2026.11.18*

> [!TIP]
> **한 줄 요약** — Diffusion sampling에 쓰이는 initial noise vector에 Fourier space에서 구조화된 패턴을 심고, 생성 후 diffusion inversion으로 noise를 복원해 그 패턴을 검출함으로써 보이지 않으면서도 robust한 watermark를 만든다.

| 흐름 | 내용 |
|:--|:--|
| **문제** | 생성 이미지의 출처를 추적하기 위한 watermark는 이미지에 보이지 않아야 하면서도, 편집·변형 공격에 견딜 수 있어야 한다 |
| **아이디어** | Initial noise의 Fourier space에 ring 형태의 key 패턴을 심어 convolution·crop·dilation·flip·rotation에 불변하도록 하고, 검출 시 DDIM inversion으로 복원한 noise에서 패턴을 확인 |
| **결과** | Stable Diffusion 등 임의의 diffusion model에 plug-in으로 적용 가능하고 FID 손실이 거의 없으며, 기존에 배포된 watermark 방식보다 훨씬 robust |

<p align="center">
  <img src="./img1.png" width="90%"><br>
  <sub><b>Figure 1.</b> Tree-Ring Watermark 개요</sub>
</p>

---

## 참고

- [Paper](https://arxiv.org/pdf/2305.20030) · 관련 리뷰: [Stable Signature](../stable-signature/README.md)
