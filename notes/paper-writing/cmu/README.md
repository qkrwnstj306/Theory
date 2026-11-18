![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=80&section=header&text=Welcome%20Paper%20Review&fontSize=50)


## How to write a paper by CMU (2026.10.04)
*Junseo Park*


***

# Paper Writing Workflow — 3개 문서 기반 Step-by-Step 정리

> **범위:** 이 문서는 사용자가 제공한 세 문서의 내용만 재구성한 체크리스트이다.  
> 외부 지식이나 추가적인 논문 작성 원칙은 포함하지 않았다.

Overleaf를 web에서 작성하지 말고 먼저, vscode extension: LaTex Workshop install > Tex Live install (검색해보기기) 하면 local에서 Latex + Claude + Codex를 같이 쓸 수 있다.

## Step 0. LLM 사용 원칙부터 정한다

- LLM을 **co-author가 아니라 reviewer simulator**로 사용한다.
- AI에게 논문 전체를 한 번에 다시 쓰게 하지 않는다.
- 한 번에 하나의 문제를 확인하고, 사람이 검토한 뒤 반영한다.
- 작업은 **section-by-section**으로 진행한다.
  - Abstract
  - Introduction
  - Method overview
  - Experiments
  - Analysis
  - Limitations
- 모호하게 `Improve this paper`라고 하지 말고 역할과 검사 항목을 구체적으로 지정한다.
  - logical flow
  - weak arguments
  - clarity
  - transitions
  - contribution distinction
- 수정할 때는 다음 조건을 명시한다.
  - `Without changing meaning`
  - `Do not add new claims, results, equations, or citations`
- 한 번에 완성하려 하지 않고 반복한다. `(Opus로 "어떻게 개선할지" 찾고, ChatGPT 계열로 실제 paragraph-level change를 수행)`
  - feedback → revise → feedback
- awkward sentence는 여러 alternative phrasing을 받아 사람이 선택한다.
- 최종 polishing에서는 **before → after** 형태의 diff-style edit을 사용한다.
- 목표 venue를 명시한다.
- LLM을 citation 생성에 사용하지 않는다.
  - related article 후보 탐색
  - related work 요약
  - prior work와의 차이 분석에는 사용할 수 있다.


- ChatGPT & Claude

```
당신은 학술 논문을 검토하고 있습니다.

공동저자처럼 새로운 내용을 만들어내지 말고,
신중하고 비판적인 Reviewer의 관점에서 검토해주세요.

논문 전체 또는 Section 전체를 임의로 다시 작성하지 마세요.

다음 내용을 새롭게 추가하지 마세요.
- 새로운 주장
- 새로운 실험 결과
- 새로운 수치
- 새로운 Equation
- 새로운 Citation
- 제공되지 않은 가정이나 기술적 사실

논문의 과학적 의미를 변경하지 마세요.

먼저 문제점을 구체적으로 찾아내고,
각 문제가 왜 문제가 되는지를 설명해주세요.

제공된 원고에서 확인할 수 있는 내용만 사용해주세요.

논문 전체에서 terminology와 notation을 일관되게 유지해주세요.

Target venue:
[학회명]

문제점을 다음 기준으로 구분해주세요.
1. 논리적 문제
2. 명확성 문제
3. 일관성 문제
4. 표현 및 Presentation 문제
```

---

# Step 1. 목표 Conference/Venue와 논문의 Story를 정한다

- 독자의 배경을 고려한다.
- 목표 conference의 audience에 맞게 다음을 조정한다.
  - Introduction
  - story
  - contributions
- 특정 분야에 지나치게 한정된 story가 목표 venue와 맞는지 확인한다.
- 논문 전체에서 전달할 **중요한 message**를 정한다.
- 중요한 message는 다음 여러 section에서 반복해서 강조할 수 있다.
  - Abstract
  - Introduction
  - Experiments
- 반복할 때는 표현을 줄이거나 바꿀 수 있지만 **terminology (용어)는 동일하게 유지**한다.

- Claude

```
[학회명] Reviewer의 관점에서 이 논문을 검토해주세요.

논문을 다시 작성하지 마세요.

현재 논문의 Story가 해당 학회의 독자에게 적절한지 확인해주세요.

다음을 검토해주세요.

1. 이 논문의 핵심 메시지가 무엇인지
2. Introduction에서 그 메시지가 명확하게 전달되는지
3. 현재 Story가 Target Venue의 독자에게 적절한지
4. Contribution이 명확하게 동기부여되어 있는지
5. 특정 분야에 지나치게 한정된 Story 때문에
   Target Venue와 맞지 않아 보일 수 있는 부분이 있는지
6. 동일한 핵심 메시지가 다음 Section에서 일관되게 전달되는지
   - Abstract
   - Introduction
   - Experiments
7. 핵심 Contribution을 설명할 때 동일한 Terminology가 유지되는지

각 문제를 다음과 같이 정리해주세요.

[위치]
[문제]
[Reviewer가 혼란스러워할 수 있는 이유]
[개선 방향]

아직 문장을 직접 다시 작성하지 마세요.

논문:
[내용 붙여넣기]
```

- GPT

```
아래에서 승인된 문제점에 해당하는 부분만 수정해주세요.

논문의 과학적 의미를 변경하지 마세요.

다음을 새로 추가하지 마세요.
- Claim
- Result
- Equation
- Citation
- 새로운 가정

목표는 다음과 같습니다.

- 논문의 핵심 메시지를 더 명확하게 만들기
- [학회명]에 맞게 표현을 다듬기
- Section 간 동일한 Terminology 유지하기

가능한 한 최소한으로 수정해주세요.

다음 형식으로만 출력해주세요.

[수정 전]
...

[수정 후]
...

[수정 이유]
...

승인한 문제점:
[Claude 결과 중 실제로 수정할 내용]

원문:
[내용 붙여넣기]
```

---

# Step 2. Main Document의 구조를 만든다

- 긴 plain text를 피하고 논문을 구조화한다.
  - section
  - subsection
  - itemize / enumerate
  - table
- 세 줄 이상으로 길어지는 문장은 분리할 부분이 있는지 확인한다.
- main logic을 방해하는 보충 설명이나 예외 설명은 footnote 사용을 고려한다.
- 뒤에서 논의되지 않거나 main message와 연결되지 않는 isolated sentence / paragraph를 찾는다.
- 필요 없는 isolated content는 제거한다.
- 새로운 내용을 설명할 때 이전 section, equation 등을 적극적으로 참조한다.
  - `\label`
  - `\ref`
  - `\eqref`
- Method 내부와 Method–Experiment 사이의 back-pointer를 점검한다.
- abbreviation은 처음 등장할 때 정의한다.
  - HMM, ASR처럼 익숙한 약어도 최초 등장 시 정의한다.

- Claude

```
다음 Section의 구조를 검토해주세요.

아직 문장을 다시 작성하지 마세요.

다음을 확인해주세요.

1. 지나치게 긴 문장
2. 긴 Plain Text 때문에 구조화가 필요한 부분
3. 논문의 핵심 메시지와 연결되지 않는 문장이나 Paragraph
4. 반복되는 설명
5. Main logic을 방해하여 Footnote로 옮기는 것이 적절한 부수적 설명
6. 이전 내용을 다시 참조하면 좋은 위치
   - Section
   - Equation
   - Figure
   - Table
7. 처음 등장할 때 정의되지 않은 Abbreviation

각 문제를 다음 형식으로 정리해주세요.

[위치]
[문제 유형]
[현재 문제]
[추천 조치]

새로운 내용을 만들지 마세요.

Section:
[내용 붙여넣기]
```

- GPT

```
아래에서 승인된 문제점만 수정해주세요.

의미를 변경하지 마세요.

새로운 Claim, Result, Equation, Citation을 추가하지 마세요.

다음을 수행해주세요.

- 너무 긴 문장을 필요한 경우 분리
- 확인된 중복 설명 제거
- Paragraph 구조 개선
- 기존 Technical Terminology 유지
- 모든 수치 그대로 유지

최소한으로 수정해주세요.

다음 형식으로 출력해주세요.

[수정 전]
...

[수정 후]
...

승인된 수정 사항:
[내용]

원문:
[내용]
```

---

# Step 3. Related Work를 정리한다

- 관련 연구를 논문 하나씩 단순 나열하지 않는다.
- 여러 related papers를 성격에 따라 한 paragraph로 묶는다.
- 각 paragraph 마지막에 **기존 연구와 proposed method의 차이**를 명확하게 쓴다.
- AI assistant와 Google Scholar를 이용해 관련 article 후보를 찾을 수 있다.
- article을 찾을 때는 필요한 조건을 구체적으로 지정한다.
- AI가 만들어낸 citation을 그대로 사용하지 않는다.

- Claude

```
다음 Related Work Section을 검토해주세요.

아직 다시 작성하지 마세요.

다음을 확인해주세요.

1. 기존 연구가 논문별로 단순 나열되어 있는 부분이 있는지
2. 여러 관련 논문을 하나의 연구 그룹 또는 유형으로 묶을 수 있는지
3. 각 Paragraph에서 기존 연구와 Proposed Method의 차이가 명확한지
4. Paragraph 마지막에 Proposed Method와의 명확한 차이가 빠져 있는지
5. Prior Work와 Proposed Method에 사용하는 Terminology가 일관적인지
6. 기존 연구에 대한 설명 중 Citation을 사람이 다시 확인해야 하는 부분이 있는지

각 Paragraph에 대해 다음 형식으로 알려주세요.

[Paragraph]
[현재 역할]
[문제]
[부족한 차별점]
[개선 방향]

새로운 Citation을 만들지 마세요.

제공되지 않은 논문의 특징을 추측하거나 만들어내지 마세요.

Related Work:
[내용 붙여넣기]
```

- GPT

```
다음 Related Work Paragraph를 다듬어주세요.

기존 연구에 대한 새로운 정보를 추가하지 마세요.

현재 Paragraph 안에 존재하는 정보만 사용해주세요.

목표:
- 기존 연구들을 그룹화한 설명 유지
- 기존 연구와 Proposed Method의 차이를 명확하게 표현
- 간결하게 작성

Citation을 추가하거나 만들어내지 마세요.

과학적 의미를 변경하지 마세요.

서로 다른 표현 3개를 제안해주세요.

원문:
[Paragraph]
```

---

# Step 4. Method와 Equation을 작성하고 점검한다

- 필요하다면 basic equation도 명시한다.
- algorithm 설명이 필요한 경우 `algorithm.sty` 사용을 고려한다.
- 서로 다른 variable에 같은 문자를 중복 사용하지 않는다.
- 변수의 type을 notation으로 구분한다.
  - scalar
  - vector
  - matrix
  - sequence
- 새로운 variable을 처음 소개할 때 domain을 정의한다.
- variable domain을 이용해 equation consistency를 확인한다.
- set과 element 등 서로 다른 domain의 항이 잘못 같아지지 않는지 확인한다.
- 설명용 superscript는 non-italic 표기를 사용한다.
- operation 이름은 non-italic으로 표기한다.
  - `max`
  - `sigmoid`
  - `BiLSTM`
- equation만 나열하지 말고 equation에 대한 discussion을 붙인다.
- equation 직후 불필요한 paragraph break를 피한다.
- equation에 오류가 없는지 별도로 매우 주의해서 검토한다.
- AI에게 다음을 별도로 검사시킬 수 있다.
  - undefined variables / operations
  - duplicate symbols
  - inconsistent variable domains
  - equation inconsistencies
  - symbol conflicts
  - missing definitions

- Claude

```
다음 Method Section에 대해 Equation과 Notation을 매우 꼼꼼하게 검토해주세요.

Method 자체를 다시 작성하지 마세요.

모든 Equation과 Variable을 하나씩 확인해주세요.

다음을 찾아주세요.

1. 정의되기 전에 사용된 Variable
2. 설명되지 않은 Operation
3. 서로 다른 Variable에 동일한 문자를 사용한 경우
4. 다음 Type 사이의 Notation이 일관되지 않은 경우
   - Scalar
   - Vector
   - Matrix
   - Sequence
5. 처음 등장했을 때 Domain이 정의되지 않은 Variable
6. Equation 내부 Variable들의 Domain이 서로 맞지 않는 경우
7. Equation 좌변과 우변 사이의 불일치 가능성
8. Non-italic으로 표기해야 할 Operation
9. Section에 따라 의미가 달라지는 Symbol
10. Equation만 제시되고 충분한 설명이 없는 부분
11. Section 또는 Equation Back-pointer가 부족한 부분

각 문제를 다음 형식으로 출력해주세요.

[Equation 또는 위치]
[Symbol]
[문제]
[왜 일관되지 않는지]
[수정 방향]

새로운 Equation을 만들지 마세요.
논문에 없는 정의를 가정하지 마세요.

Method:
[내용 붙여넣기]
```

- GPT

```
아래에서 승인된 Equation / Notation 문제만 수정해주세요.

수학적 Method 자체를 변경하지 마세요.

새로운 Equation 또는 Variable을 추가하지 마세요.

모든 수학적 의미를 그대로 유지해주세요.

각 수정은 다음 형식으로 보여주세요.

[수정 전]
...

[수정 후]
...

[수정 이유]
...

승인된 문제점:
[내용]

원문:
[내용]
```

---

# Step 5. Experimental Design을 정한다

- experimental design과 benchmark selection은 senior researcher에게 확인받는다.
- 다른 baseline 또는 reference와 비교한다.
- 여러 database에서 실험하는 것을 고려한다.

- Claude

```
다음 Experimental Design을 비판적인 Reviewer의 관점에서 검토해주세요.

새로운 실험 결과를 만들어내지 마세요.

현재 논문에 이미 제시된 내용만 기준으로 확인해주세요.

다음을 확인해주세요.

1. 논문에서 이미 언급된 관련 Baseline 또는 Reference와 비교가 이루어졌는지
2. 현재 연구에 해당한다면 여러 Database에서 실험했는지
3. 각 Experiment가 논문에서 주장하는 Claim과 명확하게 연결되는지
4. Reviewer가 지적할 가능성이 있는 Experimental Design의 약점이 무엇인지

다음 형식으로 정리해주세요.

[Claim]
[Claim을 뒷받침하는 Experiment]
[Potential Weakness]
[Senior Co-author와 직접 확인해야 할 사항]

새로운 실험 결과를 만들지 마세요.

Experimental Section:
[내용 붙여넣기]
```



---

# Step 6. Tables와 Figures를 만든다

- 다른 논문이나 자신의 이전 논문의 figure를 그대로 복사하지 않는다.
- table / figure의 font size를 충분히 크게 한다.
- table / figure는 가능한 한 **self-consistent**하게 만든다.
- caption에는 필요에 따라 다음을 짧게 포함한다.
  - experimental design
  - brief discussion
- caption은 불필요하게 길어지지 않도록 1–2개의 짧은 문장 정도로 유지한다.
- 특정 논문에서 사용하는 special abbreviation (약어)은 figure/table 안에서도 설명한다.
- figure/table은 가능하면 page의 top 또는 bottom에 배치한다.
- figure에서 사용하는 notation을 equation의 notation과 일치시킨다.
- 복잡한 table은 category를 나누고 `\hline` 등을 이용해 구분한다.
- 성격이 다른 결과를 같은 category에 섞지 않는다.
- AI/ML/NLP conference에서는 visualization 또는 example을 제공한다.
- colorblind-friendly / black-and-white-friendly인지 확인한다.
- 색만으로 구분하지 않고 다음 요소도 사용할 수 있다.
  - line markers
  - bar patterns
- matplotlib 사용 시 문서에서 제시한 예:
  - `plt.style.use('tableau-colorblind10')`
- reviewer가 abstract, figures, tables만 보고 판단할 수도 있음을 고려한다.
- AI에게 다음을 물어볼 수 있다.
  - reviewer가 10초 안에 figure의 takeaway를 이해할 수 있는가?
  - 이 figure는 어떤 질문에 답하는가?
  - caption만 보고 그것이 분명한가?
  - reviewer confusion을 줄일 annotation이 있는가?

- Claude

```
이 Figure/Table을 바쁜 Reviewer의 관점에서 검토해주세요.

아직 Figure/Table을 다시 설계하지 마세요.

다음 질문에 답해주세요.

1. 이 Figure/Table은 어떤 질문에 답하고 있는가?
2. 가장 중요한 Takeaway는 무엇인가?
3. Reviewer가 약 10초 이내에 핵심을 이해할 수 있는가?
4. Caption만 읽어도 Takeaway가 명확한가?
5. Main Text를 많이 읽지 않아도 Figure/Table 자체만으로 이해 가능한가?
6. 논문 특유의 Abbreviation이 모두 설명되어 있는가?
7. Equation과 동일한 Notation을 사용하는가?
8. Font Size가 충분히 읽기 쉬워 보이는가?
9. 정보를 구분하기 위해 Color에만 의존하고 있는가?
10. Reviewer의 혼란을 줄이기 위해 Annotation 하나를 추가한다면 무엇이 필요한가?
11. 서로 다른 Category의 Table Entry가 충분히 구분되어 있는가?

각 문제점과 추천 수정 방향만 알려주세요.

새로운 실험 결과나 해석을 추가하지 마세요.
```

- GPT

```
다음 Figure/Table Caption을 다듬어주세요.

조건:

- 과학적 의미를 변경하지 마세요.
- 새로운 Result를 추가하지 마세요.
- 제공되지 않은 Interpretation을 추가하지 마세요.
- 이미 제공된 정보 안에서 Experimental Design을 간단히 설명해주세요.
- 결과가 실제로 뒷받침하는 경우에만 짧은 Discussion을 포함해주세요.
- 이 논문에서만 사용하는 Abbreviation을 설명해주세요.
- Caption을 간결하게 유지해주세요.
- 가능하면 1~2개의 짧은 문장으로 작성해주세요.

3개의 대안을 제안해주세요.

현재 Caption:
[내용]

검증된 Figure/Table 정보:
[내용]
```

---

# Step 7. Experimental Discussion을 작성한다

- 같은 metric 안에서 decimal place를 일관되게 맞춘다.
- 결과 숫자를 단순 나열하지 않는다.
- 실험 결과가 전달하는 **story와 overall trend**를 설명한다.
- story 없이 결과만 설명하는 문장은 제거할 수 있는지 확인한다.
- experimental discussion을 이전 Method / Equation / Section과 연결한다.
  - `\ref{}`
  - `\eqref{}`
- 다른 보고보다 결과가 낮다면 실험 조건 차이나 이유가 명확하게 보이도록 한다.
  - footnote
  - table/figure remark
- AI에게 figure/table을 분석시키고 reviewer가 지적할 수 있는 weakness를 찾게 한다.

- Claude

```
다음 Experimental Discussion을 검토해주세요.

아직 다시 작성하지 마세요.

다음을 찾아주세요.

1. Table의 숫자를 단순히 반복해서 설명하는 부분
2. Overall Trend나 Message 없이 결과만 설명하는 부분
3. 같은 Metric 안에서 Decimal Place가 일관되지 않은 부분
4. 이전 Method / Equation / Section과 연결하면 좋은 Discussion
5. 다른 연구보다 결과가 낮은데 Experimental Condition의 차이가 충분히 눈에 띄지 않는 부분
6. Experiment가 보여주는 것보다 강하게 주장하는 문장
7. Table이나 Figure를 보고 Reviewer가 지적할 가능성이 있는 Weakness

각 문제를 다음 형식으로 작성해주세요.

[위치]
[문제]
[현재 Evidence가 실제로 뒷받침하는 범위]
[수정 방향]

관찰된 결과의 원인을 새롭게 만들어내지 마세요.
```

- GPT

```
다음 Experimental Result Paragraph를 다듬어주세요.

의미를 변경하지 마세요.

논문에서 이미 설명되지 않은 Causal Explanation을 새로 추가하지 마세요.

새로운 Claim, Result, Equation, Citation을 추가하지 마세요.

목표:

- 숫자를 그대로 반복하는 설명을 줄이기
- 현재 결과에서 확인되는 Overall Trend를 더 명확하게 표현하기
- 모든 숫자를 정확하게 유지하기
- Limitation과 불확실성을 그대로 유지하기
- 필요한 경우 기존 Section / Equation Reference를 활용하기

다음 형식으로 출력해주세요.

[수정 전]
...

[수정 후]
...

원문:
[내용]
```

---

# Step 8. Main Paper와 Appendix를 분리한다

- Appendix는 supplementary document이다.
- reviewer가 Appendix를 반드시 읽는다고 가정하지 않는다.
- main document 자체가 self-consistent해야 한다.
- 핵심 정보가 Appendix에만 있어서는 안 된다.
- 논문을 이해하거나 핵심 주장을 정당화하는 데 필요한 정보는 main document에 넣는다.

Appendix에 둘 수 있는 내용:

- detailed configurations, 특히 training 관련 설정
- database details
- full derivations
  - main document에는 필요한 결과나 equation만 남긴다.
- 핵심 설명에 필수적이지 않은 일부 ablation studies
- 추가적인 prompt examples
- audio spectrograms
- recognition results
- 추가 examples

대표적인 핵심 example은 main document에도 일부 포함한다.

- Claude

```
논문이 Appendix에 과도하게 의존하고 있는지 확인해주세요.

Main Paper에서 Appendix를 참조하는 모든 부분을 찾아주세요.

각 Appendix 내용이
Main Paper 자체를 이해하고 논리를 완성하는 데 필수적인지 판단해주세요.

다음 두 가지로 분류해주세요.

A. Main Document에 반드시 있어야 하는 핵심 정보
B. Appendix에 적절한 Supplementary Information

Appendix에 둘 수 있는 내용의 예:

- 상세 Training Configuration
- Database Detail
- Full Derivation
- 핵심 설명에 필수적이지 않은 추가 Ablation Study
- 추가 Prompt Example
- Audio Spectrogram
- Recognition Result
- 추가 Example

논문의 핵심 주장이나 이해에 필요한 정보가
Appendix에만 존재하는지 특히 확인해주세요.

다음 표로 정리해주세요.

| 위치 | Appendix 내용 | Main에 필수인가? | 판단 이유 |
```

---

# Step 9. References를 정리한다

- 가능하면 25개 이상의 reference를 포함하는 것을 고려한다.
- 여러 research group의 reference를 균형 있게 포함한다.
- 특정 연구 그룹이나 big tech에 reference가 편향되지 않았는지 확인한다.
- competitor의 reference도 포함한다.
- 자신의 또는 동료의 paper를 cite할 수 있지만 double-blind에서는 주의한다.
- Google Scholar BibTeX를 그대로 사용하지 말고 정리한다.
- conference name 등의 format을 normalize한다.
- title capitalization을 확인한다.
  - 예: `HMM` → `{HMM}`
- 가능하면 arXiv version보다 published version을 cite한다.
- Wikipedia보다 published paper를 cite한다.
- 신뢰할 수 있는 source를 사용한다.
  - known venues
  - known authors
- suspicious journal이나 불명확한 source는 피한다.
- 공간 절약이 필요하면 venue style에 맞춰 `et al.` 사용을 고려한다.

- Claude

```
제공된 Reference List만 검토해주세요.

새로운 Reference를 생성하지 마세요.

다음을 확인해주세요.

1. Reference Format이 일관되지 않은 부분
2. Conference Name 표기가 일관되지 않은 부분
3. 정리되지 않은 BibTeX Entry
4. Title의 Acronym Capitalization을 보호해야 할 가능성이 있는 부분
5. arXiv Version으로 인용되어 있어 Published Version 존재 여부를 사람이 확인해야 할 Reference
6. 특정 Research Group 또는 Company에 지나치게 편중되어 있는지
7. 논문 본문에서 이미 논의했지만 Reference List에서 누락된 Competitor Reference가 있는지
8. 실제 존재 여부를 사람이 다시 확인해야 할 Reference

다음 형식으로 출력해주세요.

[Reference]
[문제]
[사람이 직접 확인해야 할 내용]

Reference List:
[붙여넣기]
```

---

# Step 10. LLM을 이용해 Section-by-Section Review를 수행한다

각 section을 한 번에 하나씩 검토한다.

권장 순서:

1. Abstract
2. Introduction
3. Method overview
4. Experiments
5. Analysis
6. Limitations

각 section에서 AI에게 구체적인 역할을 준다.

예:

- reviewer로 review
- logical flow 확인
- weak argument 찾기
- clarity 확인
- prior work와 contribution 차이 확인
- redundant description 찾기
- compression 제안

Reviewer-style scoring을 사용할 수 있다.

다음 항목을 1–10점으로 평가하고 이유를 설명하게 한다.

- clarity
- novelty
- rigor
- presentation

가능하면 실제 reviewer guideline을 같이 제공한다.

- Claude: 권장 순서눈 section 별로, Abstract > Introduction > Method Overview > Experiments > Analysis > Limitations

```
[학회명] Reviewer의 관점에서 이 Section을 검토해주세요.

Section을 다시 작성하지 마세요.

다음 항목을 각각 10점 만점으로 평가해주세요.

- Clarity
- Novelty
- Rigor
- Presentation

각 점수에 대해:

1. 왜 이 점수를 주었는지
2. 왜 더 높은 점수를 주지 않았는지
3. 어떤 문장이 점수를 낮추고 있는지

를 구체적으로 설명해주세요.

그 다음 다음 문제를 찾아주세요.

1. Logical Flow 문제
2. Weak Argument
3. 모호한 표현
4. 반복 설명
5. 과도한 Claim
6. 이전 Section과의 연결 부족
7. Terminology 불일치
8. Reviewer가 혼란스러워할 가능성이 높은 부분

가장 중요한 문제부터 순서대로 정리해주세요.

새로운 내용을 추가하지 마세요.

Section:
[내용 붙여넣기]
```

---

# Step 11. Specialized Audit를 각각 별도로 수행한다

한 번에 모두 하지 말고 필요한 검사를 분리한다.

- sentence splitting pass
- abbreviation audit
- cross-reference audit
- equation symbol audit
- caption clarity rewrite
- related-work contrast rewrite
- reviewer confusion prediction
- claim overstatement detection
- appendix safety check
- terminology inconsistency check
- notation drift check
- typo check
- redundant description check
- double-blind check
- reference cleaning
- reference format check
- arXiv → publication check

- Claude

긴 문장 검사

```
논문에서 지나치게 긴 문장을 모두 찾아주세요.

특히 약 3줄 이상으로 길어지는 문장을 확인해주세요.

각 문장에 대해:

- 원문
- 왜 읽기 어려운지
- 어디에서 문장을 분리하면 좋은지

를 알려주세요.

아직 직접 다시 작성하지 마세요.
```


Abbreviation Audit

```
논문에 등장하는 모든 Abbreviation을 확인해주세요.

각 Abbreviation이 처음 등장하는 위치에서
정의되어 있는지 확인해주세요.

처음 등장할 때 정의되지 않은 Abbreviation만 목록으로 정리해주세요.
```

Cross-reference Audit

```
논문에서 Section, Equation, Figure, Table Back-pointer를 추가하면
읽기 쉬워질 위치를 찾아주세요.

특히 다음 연결을 중점적으로 확인해주세요.

- Method Section 내부
- Method → Experiment
- Experiment → 이전 Method / Equation

각 위치와 어떤 내용을 참조하면 좋은지 알려주세요.
```

Equation Symbol Audit

```
다음 문제를 모두 찾아주세요.

- 정의되지 않은 Variable
- 정의되지 않은 Operation
- 동일한 Symbol의 중복 사용
- Notation Drift
- Variable Domain 불일치
- Equation 사이의 Consistency 문제

각 문제의 정확한 위치를 알려주세요.

새로운 Equation을 만들지 마세요.
```

Claim Overstatement

```
현재 논문에서 Evidence가 실제로 보여주는 것보다
강하게 표현된 문장을 찾아주세요.

각 문장에 대해:

[원문]
[현재 Supporting Evidence]
[왜 과도한 Claim일 수 있는지]

를 설명해주세요.

새로운 Claim을 제안하지 마세요.
```

Reviewer Confusion Prediction

```
Reviewer가 가장 혼란스러워할 가능성이 높은 부분을 찾아주세요.

각 부분에 대해:

[원문]
[예상되는 오해]
[어떤 표현 또는 정보 부족 때문에 오해가 발생하는지]

를 설명해주세요.

아직 문장을 다시 작성하지 마세요.
```

Appendix Safety Check

```
Main Paper의 논리를 이해하거나 핵심 Claim을 정당화하는 데 필요한 정보가
Appendix에만 존재하는 부분을 찾아주세요.

Reviewer가 Appendix를 읽지 않는다고 가정하고 평가해주세요.
```


---

# Step 12. Submission 직전 최종 점검을 한다

- AI에게 reviewer 역할을 부여해 final pass를 수행한다.
- 목표 venue의 reviewer guideline을 함께 제공할 수 있다.
- submission guideline을 직접 확인한다.
- conference가 요구하는 추가 문서를 확인한다.
  - reproducibility
  - ethics
  - limitation
  - broad impact
  - IRB approval, if needed
- PDF 안에서 double-blind policy를 위반하는 표현을 찾는다.
- reference가 실제로 존재하는지 source와 함께 확인한다.
- terminology와 notation consistency를 다시 확인한다.
- typo를 확인한다.
- 모든 AI 수정 결과는 사람이 직접 검토한다.

- GPT

```
다음 Section을 최종적으로 최소 수정해주세요.

과학적 의미를 절대 변경하지 마세요.

다음을 새롭게 추가하지 마세요.

- Claim
- Result
- Equation
- Citation
- 기존에 없던 Explanation

다음 항목만 확인해주세요.

1. Clarity
2. Grammar
3. Terminology Consistency
4. Notation Consistency
5. 어색한 표현
6. 불필요한 반복
7. 문장 또는 Paragraph 간 Transition

작성자의 기존 문체를 최대한 유지해주세요.

수정이 필요한 문장만 다음 형식으로 출력해주세요.

[수정 전]
...

[수정 후]
...

[수정 이유]
...

수정이 필요하지 않은 문장은 출력하지 마세요.

원문:
[내용]
```

---

# Step 13. Camera-Ready 단계에서 Acknowledgment를 복구한다

- co-author, 특히 supervisor에게 필요한 acknowledgment가 있는지 확인한다.
- double-blind submission 단계에서는 허용되지 않는 acknowledgment 정보를 camera-ready에서 복구한다.
- PSC / NCSA를 사용한 경우 문서에 제시된 acknowledgement 문구를 확인해 추가한다.

- Claude

```
이 논문을 [학회명] Reviewer의 관점에서
Submission 직전 최종 검토해주세요.

논문을 다시 작성하지 마세요.

Reviewer Guideline:
[공식 Reviewer Guideline 붙여넣기]

다음을 모두 확인해주세요.

1. Clarity
2. Novelty
3. Rigor
4. Presentation
5. Logical Flow
6. Contribution의 명확성
7. Terminology Consistency
8. Notation Consistency
9. Equation Consistency
10. Abbreviation Definition
11. Figure / Table Clarity
12. Caption Clarity
13. Cross-reference
14. Claim Overstatement
15. Appendix 의존성
16. Double-blind 위반 가능성
17. Reference 문제
18. 지나치게 긴 문장
19. Typo
20. 제공된 Submission Guideline에서 요구하는 항목 중 누락된 것이 있는지

모든 문제를 다음 형식으로 정리해주세요.

[심각도]
Critical / Major / Minor

[위치]
Section, Equation, Figure 또는 정확한 문장

[문제]
Reviewer가 왜 문제로 볼 수 있는지

[추천 조치]
저자가 어떤 부분을 확인하거나 수정해야 하는지

새로운 과학적 내용을 만들어내지 마세요.
논문 전체를 다시 작성하지 마세요.
```

---

# 최종 반복 흐름

```text
Authorship / Policy 확인
        ↓
Venue / Audience / Story 정리
        ↓
Main document 구조 작성
        ↓
Related Work 정리
        ↓
Method / Equation 작성 및 검증
        ↓
Experimental Design
        ↓
Tables / Figures 작성
        ↓
Experimental Discussion
        ↓
Main vs Appendix 분리
        ↓
References 정리
        ↓
Section-by-Section LLM Review
        ↓
Specialized Audits
        ↓
Submission Guideline / Double-Blind / Reference Final Check
        ↓
Camera-Ready Acknowledgment
```

## LLM 작업의 기본 반복

```text
문제 찾기
→ 사람이 확인
→ 필요한 부분만 수정
→ 다시 review
→ final consistency check
```

## 세 문서에서 반복적으로 강조되는 원칙

- 논문 전체를 AI에게 한 번에 rewrite시키지 않는다.
- AI는 reviewer/checker처럼 사용한다.
- section-by-section으로 확인한다.
- 의미를 바꾸지 않는 최소 수정이 안전하다.
- AI가 새로운 claim, result, equation, citation을 추가하지 못하게 한다.
- 관련 연구와 proposed method의 차이를 명확하게 한다.
- equation, notation, abbreviation, cross-reference consistency를 별도로 검사한다.
- figure와 table만으로도 핵심 message가 전달되도록 한다.
- main paper는 Appendix 없이도 self-consistent해야 한다.
- reference의 실제 존재 여부와 publication version을 확인한다.
- AI의 출력은 항상 사람이 검토한다.


## 각 model에 기본적으로 넣는 .md

- Claude

```
# Academic Paper Writing Instructions

이 프로젝트는 학술 논문 작성 및 검토 프로젝트이다.

논문과 관련된 작업을 수행할 때는 반드시 다음 지침을 따른다.

@paper-writing-guide.md

## 기본 역할

논문을 검토할 때는 공동저자처럼 새로운 내용을 만들어내기보다
신중하고 비판적인 reviewer의 관점에서 검토한다.

사용자가 명시적으로 rewrite를 요청하지 않는 한,
논문 전체 또는 section 전체를 임의로 다시 작성하지 않는다.

먼저 문제를 찾고 설명한 뒤,
사용자가 수정 여부를 판단할 수 있도록 한다.

## 절대 추가하지 말아야 할 것

제공된 논문에 존재하지 않는 다음 내용을 임의로 추가하지 않는다.

- 새로운 scientific claim
- 새로운 experimental result
- 새로운 numerical value
- 새로운 equation
- 새로운 citation
- 새로운 assumption
- 제공되지 않은 technical fact

논문의 scientific meaning을 임의로 변경하지 않는다.

## 수정 원칙

문장을 수정할 때는 다음 원칙을 따른다.

- Without changing meaning.
- Do not add new claims, results, equations, or citations.
- 가능한 한 minimal edit를 수행한다.
- 저자의 기존 terminology를 유지한다.
- 저자의 기존 notation을 유지한다.
- 저자의 writing style을 가능한 한 유지한다.

문장을 수정할 때 가능하면 다음 형태를 사용한다.

수정 전:
...

수정 후:
...

수정 이유:
...

## Reviewer Mode

논문 검토 요청에서는 우선 다음을 확인한다.

1. logical flow
2. weak arguments
3. clarity
4. terminology consistency
5. notation consistency
6. overstatement
7. reviewer confusion
8. redundancy

필요한 경우 다음 항목을 1–10점으로 평가한다.

- Clarity
- Novelty
- Rigor
- Presentation

각 점수에는 반드시 근거를 제시한다.

## Section-by-Section

논문 전체를 한 번에 rewrite하지 않는다.

가능하면 다음 단위로 검토한다.

1. Abstract
2. Introduction
3. Method overview
4. Experiments
5. Analysis
6. Limitations

## Related Work

Related Work에서는:

- 기존 논문을 단순 나열하지 않는다.
- 관련 연구들을 적절한 그룹으로 묶는다.
- 각 paragraph에서 proposed method와 prior work의 차이가 명확한지 확인한다.
- 존재하지 않는 citation을 생성하지 않는다.
- citation의 실제 존재 여부를 추측하지 않는다.
- 사용자가 요청하지 않는 한 citation을 임의로 추가하지 않는다.

## Equation / Notation

Method와 Equation을 검토할 때는 다음을 확인한다.

- 정의되지 않은 variable
- 정의되지 않은 operation
- 동일 symbol의 중복 사용
- scalar / vector / matrix / sequence 표기
- variable domain 누락
- equation domain inconsistency
- notation drift
- equation 설명 부족
- section/equation back-pointer 부족

Equation 자체를 임의로 변경하지 않는다.

## Experimental Results

Experimental Results에서는:

- table의 숫자를 단순 반복하는 설명을 찾는다.
- overall trend와 message가 있는지 확인한다.
- evidence보다 강한 claim을 찾는다.
- decimal-place consistency를 확인한다.
- earlier section/equation과 연결 가능한 부분을 확인한다.

관찰된 결과의 원인을 evidence 없이 새롭게 만들어내지 않는다.

## Figures and Tables

Figure/Table을 검토할 때는 다음을 확인한다.

- reviewer가 약 10초 안에 takeaway를 이해할 수 있는가?
- 어떤 질문에 답하는 figure/table인지 명확한가?
- caption만으로 의미가 충분히 전달되는가?
- self-consistent한가?
- special abbreviation이 설명되어 있는가?
- equation과 notation이 일치하는가?
- font가 충분히 큰가?
- color에만 의존하지 않는가?
- 추가 annotation이 reviewer confusion을 줄일 수 있는가?

## Appendix

Main paper는 Appendix 없이도 self-consistent해야 한다.

논문의 핵심 주장이나 이해에 필수적인 정보가
Appendix에만 존재하지 않는지 확인한다.

## References

Reference 작업에서는:

- citation을 hallucinate하지 않는다.
- 실제 존재 여부를 확인하지 않고 단정하지 않는다.
- arXiv가 publication으로 출판되었는지 확인이 필요한 경우 이를 표시한다.
- BibTeX formatting과 capitalization consistency를 점검한다.
- 특정 research group에 reference가 편중되어 있는지 확인한다.

## Final Review

Submission 직전에는 별도 pass로 다음을 확인한다.

- sentence splitting
- abbreviation audit
- cross-reference audit
- equation symbol audit
- caption clarity
- related-work contrast
- reviewer confusion
- claim overstatement
- appendix safety
- terminology inconsistency
- notation drift
- typo
- double-blind
- reference formatting

## 핵심 원칙

LLM은 논문의 아이디어를 대신 만들어내는 co-author가 아니라
reviewer/checker로 사용한다.

가장 중요한 개선 대상은
clarity, framing, structure이다.

모든 AI output은 최종적으로 사람이 확인한다.
```

- GPT

```
이 프로젝트는 학술 논문 작성 및 검토를 위한 프로젝트다.

Project Sources에 포함된 Paper Writing Guidelines를
논문 관련 작업의 기본 기준으로 사용한다.

## 기본 역할

논문 검토 시 공동저자처럼 새로운 scientific content를 만들어내지 말고,
신중하고 비판적인 reviewer/checker의 역할을 우선한다.

사용자가 명시적으로 요청하지 않는 한
논문 또는 section 전체를 임의로 rewrite하지 않는다.

먼저 문제를 찾아 설명하고,
그 다음 사용자가 승인한 부분만 수정한다.

## 금지 사항

제공된 manuscript에 존재하지 않는 다음 내용을 임의로 추가하지 않는다.

- 새로운 claim
- 새로운 experimental result
- 새로운 numerical value
- 새로운 equation
- 새로운 citation
- 새로운 assumption
- 새로운 technical fact

Scientific meaning을 변경하지 않는다.

## 수정 원칙

문장을 수정할 때:

- 의미를 변경하지 않는다.
- 새로운 claim/result/equation/citation을 추가하지 않는다.
- minimal edit를 우선한다.
- author voice를 가능한 한 유지한다.
- terminology와 notation을 유지한다.
- 가능한 경우 수정 전 → 수정 후 형태로 보여준다.

## 기본 Review 항목

논문 검토 시 다음을 우선 확인한다.

- logical flow
- weak arguments
- clarity
- terminology consistency
- notation consistency
- redundancy
- claim overstatement
- reviewer confusion

필요하면 다음을 각각 1–10점으로 평가하고 이유를 설명한다.

- Clarity
- Novelty
- Rigor
- Presentation

## 작업 방식

논문 전체를 한 번에 처리하기보다 section-by-section으로 검토한다.

권장 순서:

Abstract
→ Introduction
→ Method overview
→ Experiments
→ Analysis
→ Limitations

## Method / Equation

다음을 검사한다.

- undefined variable / operation
- duplicate symbol
- scalar/vector/matrix/sequence notation
- missing variable domain
- domain inconsistency
- notation drift
- missing equation discussion
- missing back-pointer

## Related Work

- 논문을 단순 나열하는 구조를 찾는다.
- 관련 연구를 적절히 grouping할 수 있는지 확인한다.
- 각 paragraph에서 proposed method와 prior work의 차이를 확인한다.
- 존재하지 않는 citation을 만들지 않는다.

## Experimental Results

- 숫자의 단순 나열을 찾는다.
- overall trend/message가 있는지 확인한다.
- evidence보다 강한 claim을 찾는다.
- decimal-place consistency를 확인한다.
- 기존 section/equation과의 연결을 확인한다.

Evidence 없이 새로운 causal explanation을 만들지 않는다.

## Figure / Table

다음을 확인한다.

- 10초 안에 takeaway가 이해되는가?
- figure/table이 어떤 질문에 답하는가?
- caption이 이를 분명히 설명하는가?
- self-consistent한가?
- abbreviation이 설명되어 있는가?
- notation이 본문과 일치하는가?
- color에만 의존하지 않는가?
- reviewer confusion을 줄일 annotation이 필요한가?

## Appendix

Main paper가 Appendix 없이도 self-consistent한지 확인한다.
핵심 정보가 Appendix에만 존재하는지 검사한다.

## References

Citation을 hallucinate하지 않는다.
새 citation을 임의로 생성하지 않는다.

다음을 검사한다.

- format
- capitalization
- conference-name consistency
- arXiv → published version 확인 필요 여부
- reference balance

## Final Pass

Submission 직전에는 별도 pass로 다음을 검사한다.

- sentence splitting
- abbreviation
- cross-reference
- equation symbols
- caption clarity
- related-work contrast
- reviewer confusion
- claim overstatement
- appendix safety
- terminology
- notation
- typo
- double-blind
- references

모든 AI output은 사람이 최종 확인해야 한다.
```