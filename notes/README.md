<!-- 이 파일은 scripts/build.py 가 catalog.yml 에서 자동 생성합니다. 직접 수정하지 말고 catalog.yml 을 고치세요. -->
# 📒 Notes

## 관심 분야 정리

> 최근 / 거쳐 간 관심 분야를 넓게 정리한 글

- [Score-based Generative Models](topics/score-based-generative-models/README.md) · [PPT: Generative Modeling by Estimating Gradients of the Data Distribution](topics/score-based-generative-models/ncsn-slides.pptx) · [PDF: NCSN 4.3 inference algorithm 분석](topics/score-based-generative-models/ncsn-inference-analysis.pdf) · [Denoising Score Matching tech report](topics/score-based-generative-models/denoising-score-matching-techreport.pdf)
- [Flow Matching](topics/flow-matching/README.md)
- [Generative AI](topics/generative-ai/README.md)
- [Virtual Try-On (VITON)](topics/virtual-try-on/README.md)
- [Voice Conversion](topics/voice-conversion/README.md)

## Generative Model Theory

> 완벽한 DDPM 수식 분해는 papers/diffusion/ddpm 의 PDF 참고

- [Energy-based Model: Normalizing Constant](generative-theory/energy-based-model/README.md)
- [MCMC: Markov Chain Monte Carlo](generative-theory/mcmc/README.md)
- [SDE: Stochastic Differential Equation](generative-theory/sde/README.md)
- [Multivariate Gaussian Distribution in Generative Models](generative-theory/multivariate-gaussian/README.md) · [PPT: pixel-wise independent learning](generative-theory/multivariate-gaussian/pixel-wise-independent-learning.pptx)
- [Total Variation Loss in Generative Models](generative-theory/total-variation/README.md)
- [DDPM Denoising Process 분석](generative-theory/ddpm-denoising-process.pdf) `PDF`
- [DDPM 분석 2: Forward에서의 분산](generative-theory/ddpm-forward-variance.pdf) `PDF`
- [DDPM 분석 3: Loss 일부 분석 (L_{t-1} proof)](generative-theory/ddpm-lt-1-proof.pdf) `PDF`
- [Latent Diffusion Model Overview](generative-theory/ldm-overview.pdf) `PDF`

## Deep Learning Basics

> [Code Practice](https://github.com/qkrwnstj306/Machine-Learning-Practice)

- [1x1 kernel 의미](deep-learning/1x1-kernel.pdf) `PDF`
- [Backpropagation](deep-learning/backpropagation.pdf) `PDF`
- [CNN](deep-learning/cnn.pdf) `PDF`
- [Confusion Matrix](deep-learning/confusion-matrix.pdf) `PDF`
- [Convolutional filter about RGB](deep-learning/conv-rgb.pdf) `PDF`
- [Feature Normalization](deep-learning/feature-normalization.pdf) `PDF`
- [Gradient Descent](deep-learning/gradient-descent.pdf) `PDF`
- [Information Theory](deep-learning/information-theory.pdf) `PDF`
- [Layer와 node의 독립성](deep-learning/layer-node-independence.pdf) `PDF`
- [Loss function이 구불구불해지는 이유](deep-learning/loss-landscape.pdf) `PDF`
- [Logistic regression + Softmax](deep-learning/logistic-regression-softmax.pdf) `PDF`
- [Normalization](deep-learning/normalization.pdf) `PDF`
- [Regularization](deep-learning/regularization.pdf) `PDF`
- [Reparameterization trick](deep-learning/reparameterization-trick.pdf) `PDF`
- [Residual learning, skip connection](deep-learning/residual-learning.pdf) `PDF` · [Skip Connection PPT](deep-learning/skip-connection-slides.pptx)
- [Training, validation, test](deep-learning/train-val-test.pdf) `PDF`
- [Truncation trick](deep-learning/truncation-trick.pdf) `PDF`
- [U-Net](deep-learning/u-net.pdf) `PDF`
- [XOR problem](deep-learning/xor-problem.pdf) `PDF`
- [Weight Initialization](deep-learning/weight-initialization.pdf) `PDF`
- [RNN](deep-learning/rnn.pdf) `PDF`
- [인공지능의 이해 (혁펜하임)](deep-learning/understanding-ai-hyukppen.pdf) `PDF`

## Machine Learning

- [머신러닝 군집 & 분류 & 회귀 알고리즘](machine-learning/ch02-04-clustering-classification-regression.pdf) `PDF`
- [CNN](machine-learning/ch05-cnn.pdf) `PDF`
- [Embedding](machine-learning/ch06-embedding.pdf) `PDF`
- [NLP](machine-learning/ch07-nlp.pdf) `PDF`
- [AutoEncoder](machine-learning/ch08-autoencoder.pdf) `PDF`
- [GAN](machine-learning/ch09-gan.pdf) `PDF`
- [알고있으면 쓸모있는 AI 관련 지식](machine-learning/ch10-13-useful-ai-knowledge.pdf) `PDF`
- [Support Vector Machine (SVM)](machine-learning/svm.pdf) `PDF`

## Math & Probability

- [Bayes's Theorem: MLE & MAP](math/bayes-theorem/README.md)
- [Probability: 조건부 확률, 결합 확률](math/probability/README.md)
- [Joint / Marginal / Conditional Probability](math/joint-marginal-conditional.pdf) `PDF`
- [Gaussian Distribution](math/gaussian-distribution/README.md)
- [Taylor Series](math/taylor/README.md)
- [편미분, 전미분, 델타](math/partial-total-derivative/README.md)
- [Expectation Property](math/expectation.pdf) `PDF`
- [Jacobian Matrix](math/jacobian.pdf) `PDF`
- [라그랑주 승수법](math/lagrange.pdf) `PDF`
- [미분 공식](math/derivative-rules.pdf) `PDF`
- [분포 간 거리](math/distribution-distance.pdf) `PDF` · [KL-Divergence PPT](math/kl-divergence-slides.pptx)
- [스칼라를 벡터로 미분, 편미분](math/scalar-by-vector-derivative.pdf) `PDF`
- [중심 극한 정리](math/central-limit-theorem.pdf) `PDF`
- [평균, 분산](math/mean-variance.pdf) `PDF`
- [확률 분포](math/probability-distributions.pdf) `PDF`
- [Vector Norm](math/norm.pdf) `PDF`
- [Entropy with Information Theory](math/entropy.pdf) `PDF`
- [기초 수학 (산업 수학, 공학선형대수학, 신호와 시스템)](math/basic-math.pdf) `PDF`
- [Word Embedding using SVD (blog)](https://blog.naver.com/ckdgus1433/222030461107)

## Reinforcement Learning

> [Code Practice](https://github.com/qkrwnstj306/Reinforcement)
> OpenAI GYM == 0.22.0 근처 버전 권장 (버전마다 state 반환 형식이 다름)

- [David Silver 1~5강 정리](reinforcement-learning/david-silver-01-05.pdf) `PDF`
- [David Silver 6~7강 정리](reinforcement-learning/david-silver-06-07.pdf) `PDF`
- [강화학습 종류](reinforcement-learning/rl-taxonomy.pdf) `PDF`
- [Off-policy 분석](reinforcement-learning/off-policy.pdf) `PDF`
- [RNN with 똥 피하기 game](reinforcement-learning/rnn-dodge-game.pdf) `PDF`

## Paper Writing

- [How to Write a Paper](paper-writing/how-to-write/README.md)
- [How to Write a Paper (CMU)](paper-writing/cmu/README.md) · [AI-assisted paper writing TIPs](paper-writing/cmu/ai-assisted-paper-writing-tips.pdf) · [Paper writing TIPS with Gemini](paper-writing/cmu/paper-writing-tips-with-gemini.pdf) · [Using LLM to improve paper](paper-writing/cmu/using-llm-to-improve-paper.pdf)
