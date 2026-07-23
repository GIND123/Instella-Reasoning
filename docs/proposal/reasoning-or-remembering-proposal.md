# Reasoning or Remembering?

## Diagnosing and Attributing Reasoning Capabilities in Fully Open Language Models

**A Complete Research Proposal, Literature Survey, and Execution Blueprint**

**Principal Investigator:** Govind Arun Kumar
**Affiliation:** M.S. Applied Machine Learning, University of Maryland, College Park
**Proposed Collaborator:** Dr. Jiang Liu, Senior Applied Research Scientist, AMD GenAI
**Target Model Family:** AMD Instella (1B, 3B, 3B-Math, 3B-Long)
**Compute Constraint:** Single NVIDIA T4 (16 GB VRAM)
**Target Venue:** ICLR 2027 / NeurIPS 2027 / COLM 2027

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation and Research Gap](#2-motivation-and-research-gap)
3. [Literature Review](#3-literature-review)
   - 3.1 [LLM Reasoning Benchmarks and Evaluation](#31-llm-reasoning-benchmarks-and-evaluation)
   - 3.2 [Reasoning Consistency and Robustness](#32-reasoning-consistency-and-robustness)
   - 3.3 [Training Data Contamination in LLMs](#33-training-data-contamination-in-llms)
   - 3.4 [Training Data Attribution for LLMs](#34-training-data-attribution-for-llms)
   - 3.5 [Emergence of Reasoning Across Scale](#35-emergence-of-reasoning-across-scale)
   - 3.6 [Data-Centric AI and Training Data Quality for Reasoning](#36-data-centric-ai-and-training-data-quality-for-reasoning)
   - 3.7 [Embedding and Retrieval Infrastructure](#37-embedding-and-retrieval-infrastructure)
4. [The Instella Model Family: A Unique Opportunity](#4-the-instella-model-family-a-unique-opportunity)
   - 4.1 [Architecture and Training Details](#41-architecture-and-training-details)
   - 4.2 [Training Data Composition](#42-training-data-composition)
   - 4.3 [Model Variants](#43-model-variants)
   - 4.4 [Why Instella Is Uniquely Suited for This Study](#44-why-instella-is-uniquely-suited-for-this-study)
5. [Research Questions and Hypotheses](#5-research-questions-and-hypotheses)
6. [Proposed Methodology](#6-proposed-methodology)
   - 6.1 [Phase 1: Contamination Detection via Training Data Search](#61-phase-1-contamination-detection-via-training-data-search)
   - 6.2 [Phase 2: Baseline Reasoning Evaluation](#62-phase-2-baseline-reasoning-evaluation)
   - 6.3 [Phase 3: Consistency-Based Reliability Testing](#63-phase-3-consistency-based-reliability-testing)
   - 6.4 [Phase 4: Training Data Attribution](#64-phase-4-training-data-attribution)
   - 6.5 [Phase 5: Scale-Controlled Emergence Analysis](#65-phase-5-scale-controlled-emergence-analysis)
7. [Models, Datasets, and Tools](#7-models-datasets-and-tools)
   - 7.1 [Subject Models](#71-subject-models)
   - 7.2 [Reasoning Benchmarks](#72-reasoning-benchmarks)
   - 7.3 [Consistency Test Suites](#73-consistency-test-suites)
   - 7.4 [Training Data Sources](#74-training-data-sources)
   - 7.5 [Utility Models and Tools](#75-utility-models-and-tools)
8. [The Reasoning Reliability Atlas](#8-the-reasoning-reliability-atlas)
9. [Compute Budget and Feasibility](#9-compute-budget-and-feasibility)
10. [Timeline and Milestones](#10-timeline-and-milestones)
11. [Paper Structure](#11-paper-structure)
12. [Expected Contributions](#12-expected-contributions)
13. [Risk Analysis and Contingency Plans](#13-risk-analysis-and-contingency-plans)
14. [Target Venues and Deadlines](#14-target-venues-and-deadlines)
15. [Complete Reference List](#15-complete-reference-list)

---

## 1. Executive Summary

Large language models are evaluated primarily on accuracy: does the model get the right answer? But accuracy alone cannot distinguish genuine multi-step reasoning from sophisticated pattern matching against memorised training examples. This distinction matters profoundly — for scientific understanding, for engineering better models, and for deploying trustworthy AI systems.

This project proposes the first comprehensive study that combines three mature but disconnected research threads — reasoning benchmarks, training data contamination detection, and training data attribution — on a single fully open model family where every artifact (weights, training data, training code, data recipes) is publicly available: AMD's Instella.

The central question is deceptively simple: **when Instella solves a reasoning problem correctly, is it reasoning or remembering?**

We propose a four-stage methodology:

1. **Filter**: For every benchmark problem, search Instella's open training data for near-duplicates and paraphrases, classifying each problem as contaminated, partially contaminated, or clean.
2. **Diagnose**: Run Instella on semantically equivalent problem variants to measure not just accuracy but consistency — a novel Reliability Score that captures both correctness and robustness.
3. **Attribute**: For the most informative benchmark items, trace which training documents most influenced the model's output, revealing whether correct answers stem from generalised reasoning or memorised solutions.
4. **Prescribe**: Correlate reliability scores with training data properties to produce actionable data curation guidelines for improving reasoning.

The result is a **Reasoning Reliability Atlas** — a diagnostic map of which reasoning sub-skills are genuine, which are fragile, and what training data drives each. This is a contribution that is only possible on a fully open model like Instella, and it directly informs the next generation of the Instella training pipeline.

The entire study is designed to run on a single NVIDIA T4 GPU (16 GB VRAM), requiring no paid APIs and no external LLMs beyond Instella itself.

---

## 2. Motivation and Research Gap

### 2.1 The Accuracy Illusion

Modern LLMs achieve impressive scores on reasoning benchmarks. Instella-3B-Math, for example, shows strong performance on GSM8K and MATH. But a growing body of evidence suggests these numbers may overstate genuine reasoning ability.

Mirzadeh et al. (2024) demonstrated with GSM-Symbolic that adding a single irrelevant clause to GSM8K problems causes performance drops of up to 65% across state-of-the-art models (arXiv:2410.05229, ICLR 2025). Li et al. (2024) showed with GSM-Plus that eight types of adversarial perturbations reveal significant brittleness (ACL 2024). These findings raise a fundamental question: are models solving problems or recognising patterns from training?

However, this question remains a hypothesis rather than a demonstrated fact. Ivanova (2024) challenged the GSM-Symbolic findings on statistical grounds, showing that for 21 out of 25 models tested, there was insufficient evidence to reject the null hypothesis that GSM8K and GSM-Symbolic performance are equal. The debate is unresolved — and resolving it requires access to training data, which most model families do not provide.

### 2.2 The Open Data Advantage

Nearly all frontier LLMs (GPT-4, Claude, Gemini, DeepSeek, Llama 3.1) keep their training data proprietary. This makes contamination analysis impossible from the outside. Researchers must resort to indirect methods: perplexity-based membership inference (Shi et al., 2024), n-gram overlap heuristics (Brown et al., 2020), or rephrased detection (Yang et al., 2023). These methods are noisy, have known failure modes, and cannot provide ground-truth attribution.

Instella is one of the very few competitive model families where the full training data is documented and accessible. This transforms the "reasoning or remembering" question from philosophical speculation into an empirically testable hypothesis.

### 2.3 The Gap This Work Fills

The existing literature has three disconnected threads:

- **Reasoning benchmarks** measure accuracy and, increasingly, robustness — but without access to training data, they cannot distinguish contamination from genuine capability.
- **Contamination detection** identifies whether benchmark items appeared in training — but does not connect this to reasoning quality or consistency.
- **Training data attribution** traces model outputs back to influential training examples — but has not been applied specifically to understanding reasoning capabilities.

No published work combines all three on a fully open model to produce a unified picture of which reasoning is real, which is memorised, and what training data drives the difference. This is the gap we fill.

---

## 3. Literature Review

### 3.1 LLM Reasoning Benchmarks and Evaluation

#### 3.1.1 Classical Foundational Benchmarks

The evaluation of mathematical and logical reasoning in LLMs has been shaped by a small number of influential benchmarks.

**GSM8K** (Cobbe et al., 2021; arXiv:2110.14168) provides 8.5K grade-school math word problems requiring 2-8 step solutions. It has become the de facto standard for evaluating basic arithmetic reasoning, though its ubiquity has raised contamination concerns.

**MATH** (Hendrycks et al., NeurIPS 2021; arXiv:2103.03874) offers 12.5K competition-level mathematics problems across seven subjects (Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, Precalculus) at five difficulty levels. Its difficulty makes it a more discriminating test of mathematical reasoning.

**ARC-Challenge** (Clark et al., 2018; arXiv:1803.05457) contains 7,787 grade-school science questions in multiple-choice format, specifically designed so that retrieval and co-occurrence methods fail. The "Challenge" split contains only questions that both retrieval and word co-occurrence baselines answer incorrectly.

**LogiQA** (Liu et al., IJCAI 2020; arXiv:2007.08124) and **LogiQA 2.0** (Liu et al., IEEE/ACM TASLP 2023) provide logical reasoning questions sourced from the Chinese National Civil Service Examination, covering categorical reasoning, sufficient conditional reasoning, necessary conditional reasoning, disjunctive reasoning, and conjunctive reasoning.

**BIG-Bench Hard (BBH)** (Suzgun et al., ACL Findings 2023; arXiv:2210.09261) curates 23 of the most challenging tasks from BIG-Bench (Srivastava et al., TMLR 2023; arXiv:2206.04615) — specifically those where language models had previously failed to surpass the average human rater. Tasks span algorithmic reasoning (e.g., multistep arithmetic), natural language understanding (e.g., snarks, disambiguation), and world knowledge.

**CLUTRR** (Sinha et al., EMNLP-IJCNLP 2019; arXiv:1908.06177) provides a diagnostic benchmark for inductive reasoning over kinship relations. Its key property is systematic compositionality: the model must infer family relationships by composing relational steps (e.g., "A is B's mother, B is C's sister" → "A is C's mother"). Crucially, CLUTRR can generate held-out rule combinations never seen during training, making it ideal for testing genuine compositional reasoning versus memorisation.

**HumanEval** (Chen et al., 2021; arXiv:2107.03374) evaluates code generation through 164 hand-written programming problems, testing algorithmic reasoning through functional correctness.

**ReClor** (Yu et al., ICLR 2020; arXiv:2002.04326) sources logical reasoning problems from standardised tests (GMAT and LSAT), requiring the ability to identify logical structures, evaluate arguments, and detect reasoning fallacies.

#### 3.1.2 Recent Reasoning Benchmarks (2024–2026)

The limitations of classical benchmarks have driven a wave of newer evaluations:

**TTT-Bench** (Mishra et al., EMNLP 2025; arXiv:2506.10209) evaluates reasoning through novel Tic-Tac-Toe-style games, designed to test strategic reasoning on problems that cannot appear in pretraining data. Notably, this benchmark was created by the AMD Instella team — Prakamya Mishra, Jiang Liu, Jialian Wu, Xiaodong Yu, Zicheng Liu, and Emad Barsoum — making it particularly relevant to our proposal.

**MindGames** (Sileo & Lernould, EMNLP Findings 2023; arXiv:2305.03353) targets Theory of Mind using dynamic epistemic modal logic, testing whether models can reason about beliefs, knowledge, and information states.

**BIG-Bench Extra Hard (BBEH)** (Kazemi et al., 2025; arXiv:2502.19187) further raises the difficulty bar beyond BBH.

**MR-GSM8K** (Zeng et al., 2023/2024; arXiv:2312.17080) introduces meta-reasoning over GSM8K solutions, requiring models to evaluate and reason about reasoning chains rather than simply generating them.

#### 3.1.3 Consistency and Robustness-Focused Benchmarks

A critical development in reasoning evaluation has been the shift from measuring accuracy alone to measuring consistency across problem variants:

**GSM-Symbolic** (Mirzadeh et al., ICLR 2025; arXiv:2410.05229) is the landmark work in this space. It generates symbolic variants of GSM8K problems by replacing numerical values and names while preserving mathematical structure. The paper's central finding — performance drops of up to 65% when irrelevant clauses are added — directly motivates our "reasoning or remembering" framing.

**GSM-Plus** (Li et al., ACL 2024; arXiv:2402.19255) applies eight types of perturbations to GSM8K: numerical variation, arithmetic complexity increase, problem understanding, distractor insertion, critical thinking, and three others. It provides a comprehensive perturbation taxonomy that we adapt for our consistency suites.

**MATH-Perturb** (Huang et al., 2025; arXiv:2502.06453) extends perturbation-based evaluation to the harder MATH benchmark, applying both surface-level and structural perturbations to competition-level problems.

**Functional Benchmarks** (Srivastava et al., 2024; arXiv:2402.19450) proposes parameterised benchmark templates that generate unlimited novel instances, enabling measurement of the "reasoning gap" — the difference between performance on original and novel instances.

**Putnam-AXIOM** (Gulati et al., 2025; arXiv:2508.08292; earlier ICML 2024 workshop version) provides functional and static variants of Putnam competition problems, targeting higher-level mathematical reasoning. Its static/functional split provides a clean test of memorisation versus reasoning at competition difficulty.

### 3.2 Reasoning Consistency and Robustness

Beyond benchmark-specific perturbation studies, a broader literature examines whether LLMs genuinely reason or exploit shortcuts:

**Embers of Autoregression** (McCoy et al., 2023; arXiv:2309.13638; PNAS 2024) provides a theoretical framework for understanding LLM "reasoning" as shaped by output probability distributions and task frequency in pretraining data. The paper demonstrates that LLM performance on reasoning tasks correlates strongly with the probability of the correct answer under the training distribution — models succeed when the right answer is also the most probable completion, and fail when it is not.

**Faith and Fate** (Dziri et al., NeurIPS 2023 Spotlight; arXiv:2305.18654) demonstrates fundamental limits of Transformers on compositional reasoning. Using multiplication and dynamic programming tasks, the authors show that Transformers tend to linearise multi-step compositional problems, succeeding on easy instances through pattern matching but failing on harder instances that require genuine composition. This finding is directly relevant to our work: it predicts that Instella should show high accuracy but low consistency on compositional reasoning tasks.

**ReasonAgain** (Yu et al., 2024; arXiv:2410.19056) proposes a methodology for re-evaluating reasoning claims in LLMs. Notably, Xiaodong Yu — a co-author — is a member of the AMD Instella team, creating a direct intellectual connection between existing work and our proposal.

**Path-of-Thoughts** (2024; arXiv:2412.17963) examines relational reasoning robustness, testing whether models can follow chains of logical relationships consistently.

### 3.3 Training Data Contamination in LLMs

#### 3.3.1 Foundational Contamination Work

The problem of benchmark contamination was recognised from the earliest large-scale LLMs. **Brown et al. (NeurIPS 2020; arXiv:2005.14165)** — the GPT-3 paper — included an appendix analysing n-gram overlap between the pretraining corpus and evaluation benchmarks, establishing the practice of contamination reporting.

**Carlini et al.** have produced the definitive work on memorisation in neural language models:

- "Extracting Training Data from Large Language Models" (USENIX Security 2021; arXiv:2012.07805) demonstrated that GPT-2 memorises and regurgitates training data, including personally identifiable information.
- "Quantifying Memorisation Across Neural Language Models" (ICLR 2023; arXiv:2202.07646) established scaling laws for memorisation: larger models memorise more, and memorisation increases with data duplication. This is directly relevant to our study — if GSM8K-like problems appear multiple times in Instella's training data, memorisation is expected.

#### 3.3.2 Detection Methods

Three families of contamination detection have emerged:

**N-gram and embedding overlap methods** compare evaluation items directly against the training corpus. This is the most reliable approach when training data is available (as it is for Instella), but it misses paraphrased contamination.

**Perplexity and loss-based membership inference** — **Shi et al. (ICLR 2024; arXiv:2310.16789)** introduced Min-K% Prob, which detects pretraining data membership by examining whether a model assigns unusually low loss to specific tokens. The method works without access to training data but has higher false-positive rates.

**Paraphrase-aware detection** — **Yang et al. (2023; arXiv:2311.04850)** demonstrated that n-gram filtering is insufficient because models can be contaminated by paraphrased versions of benchmark items. They propose rephrased evaluation as a more robust detection strategy. **Zhou et al. (2023; arXiv:2311.01964)** and **Sainz et al. (EMNLP Findings 2023)** further document the prevalence of subtle contamination.

**Fu et al. (2024; arXiv:2406.04244)** provide a comprehensive survey of benchmark contamination methods and findings across the field.

#### 3.3.3 Contamination in Reasoning Models Specifically

A particularly relevant recent finding comes from **"On the Fragility of Benchmark Contamination Detection in Reasoning Models" (2025; arXiv:2510.02386)**, which demonstrates that reinforcement learning post-training (as used in Instella-Math) can obscure evidence of contamination introduced during SFT stages. This means that for Instella-Math, attribution should focus on pre-RL checkpoints where contamination signals are more detectable.

#### 3.3.4 Open Models as Contamination Testbeds

**Pythia** (Biderman et al., ICML 2023; arXiv:2304.01373) provides a suite of models from 70M to 12B parameters trained on identical data (the Pile) with all intermediate checkpoints available. Pythia established the methodological template for controlled contamination studies on open models — we apply the same principle to Instella, which offers a more modern training pipeline and competitive performance.

### 3.4 Training Data Attribution for LLMs

#### 3.4.1 Foundational Attribution Methods

**Influence Functions** (Koh & Liang, ICML 2017 Best Paper; arXiv:1703.04730) introduced the idea of measuring how individual training examples affect model predictions by computing the change in loss when a training point is upweighted infinitesimally. The method requires computing inverse Hessian-vector products, which is prohibitively expensive for LLMs at scale.

**TracIn** (Pruthi et al., NeurIPS 2020; arXiv:2002.08484) provides a cheaper approximation by tracing the influence of training examples through gradient descent checkpoints. For a training example z and test example z', TracIn computes the dot product of gradients across saved checkpoints. The checkpoint-proximal variant (TracIn-CP) uses only the final checkpoint and is feasible for LLM-scale models when combined with pre-filtering.

**Datamodels** (Ilyas et al., ICML 2022; arXiv:2202.00622) take a different approach: train many models on random subsets of the training data and learn a linear predictor of which training examples influence each test prediction. While producing high-quality attributions, the method requires training hundreds to thousands of models, making it infeasible for LLM-scale work.

**TRAK** (Park et al., ICML 2023; arXiv:2303.14186) achieves datamodel-quality attributions at a fraction of the compute cost through random projection and a closed-form kernel estimator. TRAK is feasible for models up to several billion parameters with ~10 surrogate runs.

#### 3.4.2 Attribution at LLM Scale

**Grosse et al. (Anthropic, 2023; arXiv:2308.03296)** demonstrated the most ambitious application of influence functions to LLMs to date, scaling to models with up to 52 billion parameters using the Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EK-FAC) approximation. Their work confirmed that influence functions can produce meaningful attributions at scale, identifying training sequences that are topically relevant, stylistically similar, or contain the same knowledge as test outputs. However, the computational requirements (Anthropic's cluster) make this approach infeasible for resource-constrained settings.

**TrackStar** (Chang et al., 2024; arXiv:2410.17413) proposes scalable influence and fact tracing for LLM pretraining, aiming to bridge the gap between theoretical influence functions and practical large-scale use.

**DDA** (Wu et al., 2024; arXiv:2410.01285) enhances training data attribution by incorporating fitting error consideration, improving attribution accuracy for LLMs.

**Concept Influence** (Kowal et al., 2026; arXiv:2602.14869) is the most recent and potentially most relevant method for our proposal. It leverages interpretability tools (probes and sparse autoencoders) to perform attribution at the concept level rather than the individual-example level. This approach is an order of magnitude cheaper than influence functions and is explicitly designed to be compatible with limited-compute settings. Given our T4 constraint, Concept Influence is a strong candidate for the attribution component.

#### 3.4.3 Practical Considerations and Cautions

**Bae et al. (2022; arXiv:2209.05364)** ("If Influence Functions are the Answer, Then What is the Question?") provide important methodological guidance on when influence functions are appropriate and how to interpret their outputs.

**"Do Influence Functions Work on Large Language Models?"** (2024; arXiv:2409.19998) offers a cautionary empirical study, finding that influence function approximations can be unreliable at LLM scale without careful validation. This motivates our multi-method approach (embedding similarity + TracIn + optional TRAK/Concept Influence validation).

**Yeh et al. (NeurIPS 2022; arXiv:2202.11844)** ("First is Better Than Last for Language Data Influence") show that early-layer representations often provide better influence estimates than final-layer representations for language models — a practical insight for our TracIn implementation.

### 3.5 Emergence of Reasoning Across Scale

The question of whether reasoning "emerges" at specific model scales is directly relevant to our comparison of AMD OLMo-1B and Instella-3B.

**Wei et al. (TMLR 2022; arXiv:2206.07682)** reported that certain capabilities appear to emerge discontinuously at specific scale thresholds — models below the threshold show near-zero performance while models above show strong performance, with no smooth transition.

**Schaeffer, Miranda & Koyejo (NeurIPS 2023 Outstanding Paper; arXiv:2304.15004)** challenged this finding, demonstrating that "nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance." The apparent emergence is an artifact of metric choice, not a fundamental property of the models.

This debate is directly testable on Instella. Since AMD OLMo-1B and Instella-3B are trained on the same data pipeline, we can perform a controlled comparison that isolates scale from data effects — something impossible with model families that change training data between sizes.

**Chain-of-Thought Prompting** (Wei et al., NeurIPS 2022; arXiv:2201.11903) showed that the benefit of explicit reasoning traces interacts with model scale — smaller models often perform worse with CoT than without. Understanding this interaction on Instella-1B versus 3B is part of our emergence analysis.

**Scaling Laws for Neural Language Models** (Kaplan et al., 2020; arXiv:2001.08361) provide the theoretical backdrop for understanding how capabilities scale with model size, but focus on loss rather than task-specific reasoning abilities.

### 3.6 Data-Centric AI and Training Data Quality for Reasoning

Understanding what training data produces reasoning ability is the ultimate goal of our attribution analysis. The following works provide context:

**Minerva** (Lewkowycz et al., NeurIPS 2022; arXiv:2206.14858) demonstrated that continued pretraining on technical and mathematical content (from arXiv papers) significantly improves mathematical reasoning. This was an early demonstration that training data composition, not just scale, drives reasoning.

**Llemma** (Azerbayev et al., ICLR 2024; arXiv:2310.10631) and **OpenWebMath** (Paster et al., ICLR 2024; arXiv:2310.06786) together showed that a curated mathematical web corpus (34.9B tokens) combined with code and scientific papers produces a 7B model competitive with much larger models on mathematics. This directly informs our attribution analysis: if Instella's reasoning traces back to OpenWebMath-like data, it suggests the reasoning is learned from mathematical exposition; if it traces to near-duplicates of benchmark solutions, it suggests memorisation.

**OpenMathInstruct-2** (Toshniwal et al., 2024; arXiv:2410.01560) provides 14 million problem-solution pairs for GSM8K and MATH, generated by Llama models. Instella-Math uses this dataset in its SFT stage, creating a known pathway for benchmark contamination that our study can directly investigate.

**DCLM** (Li et al., NeurIPS 2024 D&B; arXiv:2406.11794) and **FineWeb/FineWeb-Edu** (Penedo et al., NeurIPS 2024 D&B; arXiv:2406.17557) are the primary web corpora used in Instella's pretraining. Understanding what reasoning-relevant content exists in these corpora is essential for our attribution analysis.

**MAmmoTH2** (Yue et al., NeurIPS 2024; arXiv:2405.03548) demonstrates that scaling mathematical instruction data from the web (rather than synthetic generation) can improve reasoning, providing a data-centric perspective complementary to our work.

**TinyGSM** (Liu et al., 2023; arXiv:2312.09241) shows that small models can achieve strong GSM8K performance with sufficient targeted data, raising the question of whether this performance reflects reasoning or overfitting to the problem distribution.

**SmolLM2** (Ben Allal et al., 2025; arXiv:2502.02737) provides a recent example of data-centric training for small LMs, demonstrating that careful data curation can compensate for scale limitations.

### 3.7 Embedding and Retrieval Infrastructure

The contamination detection and attribution components of our pipeline rely on efficient embedding and similarity search:

**FAISS** (Johnson, Douze & Jégou, IEEE TBD 2019; arXiv:1702.08734) provides the core infrastructure for billion-scale similarity search. The library supports multiple index types:

- **Flat (exact search):** Feasible for up to ~10M vectors on CPU with 16GB RAM.
- **IVF (inverted file index):** Enables approximate search over 100M+ vectors.
- **HNSW** (Malkov & Yashunin, TPAMI 2018; arXiv:1603.09320): Graph-based approximate nearest neighbor search, offering the best recall-speed tradeoff for our scale.
- **Product Quantization** (Jégou, Douze & Schmid, IEEE TPAMI 2011): Compresses vectors for memory-efficient storage.

The updated **FAISS library paper** (Douze et al., 2024; arXiv:2401.08281) documents recent improvements including GPU acceleration, which could be leveraged on our T4 for the search phase.

**Sentence-BERT** (Reimers & Gurevych, EMNLP-IJCNLP 2019; arXiv:1908.10084) established the paradigm of computing fixed-size sentence embeddings through siamese BERT networks. The `all-MiniLM-L6-v2` model — built on the **MiniLM** architecture (Wang et al., NeurIPS 2020; arXiv:2002.10957) — provides 384-dimensional embeddings at ~80MB model size, running comfortably on CPU.

**GTE (General Text Embeddings)** (Li et al., 2023; arXiv:2308.03281) offers higher-quality embeddings with `gte-small` (33M parameters, 384 dimensions) and `gte-large` (335M parameters, 1024 dimensions). For our scale, `gte-small` provides a good quality-compute tradeoff.

---

## 4. The Instella Model Family: A Unique Opportunity

### 4.1 Architecture and Training Details

Instella-3B is a decoder-only Transformer with:

- **Parameters:** 3 billion
- **Architecture:** 36 decoder layers, 32 attention heads, 2,560 hidden dimension
- **Context length:** 4,096 tokens (base); 128K tokens (Instella-Long)
- **Tokenizer:** OLMo tokenizer (~50K vocabulary)
- **Training hardware:** 128 AMD Instinct MI300X GPUs
- **Training data volume:** 4.15 trillion tokens
- **Training stages:** 2-stage pretraining (general corpus → domain-specific enrichment)

The model architecture adapts the OLMo framework (Groeneveld et al., ACL 2024; arXiv:2402.00838), building on AI2's fully open language model paradigm. OLMo 2 (Walsh et al., 2025; arXiv:2501.00656) provides the most recent iteration of this architecture.

### 4.2 Training Data Composition

Instella's training data is fully documented across two stages:

**Stage 1 — General Pretraining:**

- **OLMoE-mix-0924:** A curated web corpus derived from DCLM and related sources
- **DCLM (DataComp-LM):** Large-scale web data filtered for quality (Li et al., NeurIPS 2024 D&B)
- **FineWeb-Edu:** Educational web content filtered for quality (Penedo et al., NeurIPS 2024 D&B)

**Stage 2 — Domain-Specific Enrichment:**

- **dolmino-mix-1124:** A curated mixture including mathematical, scientific, and code content
- **python-edu:** Educational Python code
- **dm_math:** DeepMind's mathematical content
- **Instella-GSM8K-synthetic:** Synthetic GSM8K-style problems generated for mathematical reasoning training. This dataset is available on HuggingFace (`amd/Instella-GSM8K-synthetic`).

**Post-Training (Instella-Math):**

- **OpenMathInstruct-2:** 14 million GSM8K and MATH problem-solution pairs (Toshniwal et al., 2024)
- **RL training via VERL** on 32 MI300X GPUs through 3 stages of reinforcement learning

This documentation level is exceptional. For every benchmark problem we test, we can search the actual training data to determine whether it (or something similar) was seen during training.

### 4.3 Model Variants

The Instella family provides a controlled comparison across several dimensions:

| Model | Parameters | Special Training | Context | Key Feature |
|---|---|---|---|---|
| AMD OLMo-1B | 1B | None | 4K | Baseline, smaller scale |
| Instella-3B | 3B | None | 4K | Base model, same pipeline as 1B |
| Instella-3B-Instruct | 3B | SFT | 4K | Instruction-tuned |
| Instella-3B-Math | 3B | SFT + RL | 4K | Math reasoning specialist |
| Instella-3B-Long | 3B | Extended training | 128K | Long context |

For our study, the critical comparisons are:

- **AMD OLMo-1B vs Instella-3B:** Same data pipeline, different scale → isolates scale effects
- **Instella-3B vs Instella-3B-Math:** Same base model, added reasoning training → isolates effect of reasoning-focused post-training
- **Instella-3B-Instruct vs Instella-3B-Math:** Both post-trained, different objectives → isolates math-specific training

### 4.4 Why Instella Is Uniquely Suited for This Study

Among competitive open model families, Instella is uniquely positioned:

| Property | Instella | Llama 3.x | Qwen 2.5 | DeepSeek | Pythia | OLMo 2 |
|---|---|---|---|---|---|---|
| Open weights | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Open training data | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Open training code | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Open data recipe | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Multiple sizes, same pipeline | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Math reasoning variant | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Competitive with frontier 3B models | ✓ | N/A | ✓ | ✓ | ✗ | ✓ |

Pythia has full openness but uses older architecture and training methods, and lacks a reasoning-focused variant. OLMo 2 is comparable in openness but does not have a math reasoning variant trained with RL. Instella uniquely combines full openness, modern training, competitive performance, and a reasoning-focused variant — making it the ideal subject for this study.

---

## 5. Research Questions and Hypotheses

### RQ1: How much of Instella's benchmark performance is attributable to training data contamination?

**H1a:** A significant fraction (>15%) of GSM8K test problems have near-duplicates (cosine similarity >0.85) in Instella's training data, particularly through the Instella-GSM8K-synthetic dataset and OpenMathInstruct-2.

**H1b:** Instella's accuracy on contaminated problems is significantly higher (>10 percentage points) than on clean problems, after controlling for difficulty.

### RQ2: Does Instella reason consistently or pattern-match?

**H2a:** On problems where Instella achieves high accuracy, consistency (measured as agreement across semantically equivalent variants) will be significantly lower for contaminated problems than for clean problems. This would indicate that high accuracy on contaminated items reflects memorisation (fragile) rather than reasoning (robust).

**H2b:** Different reasoning sub-skills (arithmetic, logical deduction, constraint satisfaction, etc.) will show different reliability profiles, with some sub-skills showing high accuracy + high consistency (genuine) and others showing high accuracy + low consistency (fragile).

### RQ3: What training data drives genuine reasoning versus memorisation?

**H3a:** For problems where Instella shows high reliability (accuracy × consistency), the most influential training data will be diverse — mathematical expositions, textbook explanations, code implementing similar algorithms — rather than near-duplicates of the problem.

**H3b:** For problems where Instella shows high accuracy but low consistency (suspected memorisation), the most influential training data will be concentrated — a small number of near-duplicate training examples.

### RQ4: How do reasoning capabilities change across scale and post-training?

**H4a:** The 1B→3B transition will show smooth improvement on sub-skills where 1B already shows partial capability, but may show apparent "emergence" on sub-skills where 1B shows near-zero performance.

**H4b:** Instella-Math's RL post-training will improve accuracy and consistency on mathematical reasoning but not on non-mathematical reasoning skills (logical deduction, constraint satisfaction), providing evidence that RL teaches task-specific reasoning rather than general reasoning ability.

### RQ5: What data curation strategies would most improve Instella's reasoning?

**H5:** For each low-reliability reasoning sub-skill, there exists a specific class of training data (identifiable through attribution analysis) whose density in the training corpus correlates with reliability. Increasing this density through targeted curation would improve both accuracy and consistency.

---

## 6. Proposed Methodology

### 6.1 Phase 1: Contamination Detection via Training Data Search

#### Objective
For every benchmark problem in our evaluation suite, determine whether Instella saw a near-duplicate, paraphrase, or structurally similar problem during training.

#### Method

**Step 1 — Embed training data:**

Sample representative passages from each of Instella's training data sources. For the large web corpora (DCLM, FineWeb-Edu), we sample 5-10 million passages. For the smaller domain-specific datasets (dm_math, Instella-GSM8K-synthetic, python-edu), we index the complete dataset. Total: approximately 10-15 million passages.

Each passage is embedded using `all-MiniLM-L6-v2` (384 dimensions) or `gte-small` (384 dimensions), producing approximately 6GB of vectors.

**Step 2 — Build FAISS index:**

Construct a FAISS index over the training data embeddings. For 10M vectors at 384 dimensions:

- Flat index (exact search): ~15 GB RAM, feasible on CPU
- IVF4096 + PQ48 (approximate): ~2 GB RAM, significantly faster
- HNSW32: ~20 GB RAM, best recall

We recommend starting with IVF + PQ for speed during development, then validating critical results with exact search.

**Step 3 — Search for each benchmark problem:**

For each of the ~8,000 benchmark problems across our evaluation suite:

```
query_embedding = embed(benchmark_problem)
top_20_neighbors = faiss_index.search(query_embedding, k=20)
for each neighbor:
    compute exact cosine similarity
    compute n-gram overlap (13-gram, following Brown et al. 2020)
    compute token-level edit distance
```

**Step 4 — Classify contamination level:**

Each benchmark problem is classified into one of three categories:

- **Contaminated (C):** Cosine similarity > 0.90 OR 13-gram overlap > 0.6. The training data contains a near-duplicate.
- **Partially contaminated (PC):** 0.75 < cosine similarity ≤ 0.90 OR 0.3 < 13-gram overlap ≤ 0.6. The training data contains a paraphrase or structurally similar problem.
- **Clean (N):** Cosine similarity ≤ 0.75 AND 13-gram overlap ≤ 0.3. No similar training data found.

**Step 5 — Manual validation:**

Sample 50 problems from each category (150 total) and manually verify the classification. Adjust thresholds if precision or recall is below 0.85.

#### Deliverable
A contamination label (C/PC/N) for every benchmark problem, with supporting evidence (the matched training example, similarity score, and overlap metrics).

### 6.2 Phase 2: Baseline Reasoning Evaluation

#### Objective
Establish Instella's performance on each benchmark, disaggregated by contamination level.

#### Method

Run each Instella variant (1B, 3B, 3B-Instruct, 3B-Math) on all benchmark problems with chain-of-thought prompting. For each problem:

1. Generate a chain-of-thought response (greedy decoding, max 1024 tokens)
2. Extract the final answer using format-specific parsing
3. Score correctness against the ground truth

Report accuracy separately for each contamination category:

```
Overall accuracy = correct / total
Contaminated accuracy = correct_C / total_C
Partially contaminated accuracy = correct_PC / total_PC
Clean accuracy = correct_N / total_N
```

The headline finding will be the accuracy gap between contaminated and clean problems. If Instella-3B scores, say, 82% on contaminated GSM8K problems but 55% on clean ones, that directly quantifies how much "reasoning" is memorisation.

#### Deliverable
Accuracy tables disaggregated by benchmark, contamination level, model variant, and reasoning sub-skill.

### 6.3 Phase 3: Consistency-Based Reliability Testing

#### Objective
Measure whether Instella's correct answers reflect genuine reasoning (consistent across equivalent formulations) or pattern matching (inconsistent across surface-level changes).

#### Method

**Step 1 — Create consistency clusters:**

For each of 500-1,000 selected benchmark problems (stratified across benchmarks, contamination levels, and difficulty), create 3-5 semantically equivalent variants through:

- **Numerical perturbation:** Change numerical values while preserving mathematical structure (following GSM-Symbolic methodology)
- **Entity substitution:** Replace names, objects, and contexts while preserving logical structure
- **Premise reordering:** Shuffle the order of given information
- **Irrelevant context injection:** Add a distracting sentence that does not change the answer
- **Rephrasing:** Express the same problem in different words

For arithmetic problems, we adapt the GSM-Symbolic template system. For logic problems, we develop new perturbation templates based on the CLUTRR compositionality framework. For constraint satisfaction problems, we permute constraint orderings.

Total: approximately 3,000-5,000 variant problems.

**Step 2 — Run Instella on all variants:**

Generate chain-of-thought responses for the original problem and all variants. Extract and score answers.

**Step 3 — Compute consistency and reliability scores:**

For each problem cluster (original + variants):

```
accuracy = fraction of cluster answered correctly
consistency = fraction of cluster where all answers agree
             (either all correct or all incorrect)

Reliability Score = accuracy × consistency
```

This Reliability Score is our novel metric. It captures both correctness and robustness:

| Accuracy | Consistency | Reliability | Interpretation |
|---|---|---|---|
| High | High | High | **Genuine reasoning** — model understands the problem structure |
| High | Low | Low | **Fragile pattern matching** — model recognises the original but fails variants |
| Low | High | Low | **Consistent inability** — model lacks the skill entirely |
| Low | Low | Very Low | **Random behavior** — model has no grasp of the problem type |

**Step 4 — Cross-reference with contamination:**

The key analysis: compute Reliability Scores separately for contaminated and clean problems. If contaminated problems show HIGH accuracy but LOW consistency while clean problems show LOWER accuracy but HIGHER consistency, this is strong evidence that benchmark contamination inflates accuracy without building genuine reasoning.

#### Deliverable
Reliability Scores for every tested problem, disaggregated by reasoning sub-skill and contamination level. The Reasoning Reliability Atlas visualisation (see Section 8).

### 6.4 Phase 4: Training Data Attribution

#### Objective
For the most informative benchmark problems, trace which training documents most influenced Instella's output, revealing the mechanistic basis of correct (and incorrect) answers.

#### Method

We employ a tiered attribution strategy, from cheapest to most expensive:

**Tier 1 — Embedding-based attribution (all problems):**

Using the FAISS index from Phase 1, the top-20 nearest training documents for each benchmark problem serve as a rough attribution. For correct answers, we characterise these neighbours:

- Are they near-duplicates of the benchmark problem? → memorisation signal
- Are they thematically related but structurally different (e.g., different math problems using similar concepts)? → generalisation signal
- Are they from a specific data source (OpenWebMath, code, encyclopedic text)? → data source attribution

**Tier 2 — TracIn-CP (selected problems, ~500):**

For the most interesting problems (high accuracy + low consistency, or high reliability on clean problems), we compute TracIn-CP attributions:

1. Load Instella-3B checkpoint
2. For each candidate training example (pre-filtered by FAISS top-100):
   - Compute gradient of loss on the training example
   - Compute gradient of loss on the test example
   - TracIn score = dot product of gradients
3. Rank training examples by TracIn score

Following the insight from Yeh et al. (NeurIPS 2022), we compute gradients at early-to-middle layers rather than the final layer.

To manage compute on T4: we use gradient checkpointing, process in micro-batches, and pre-filter with FAISS to limit the candidate set to ~100 training examples per test problem.

**Tier 3 — Concept Influence (validation subset, ~100 problems):**

For a small validation subset, we apply Concept Influence (Kowal et al., 2026) using probes trained on Instella's internal representations. This provides a complementary attribution signal that operates at the concept level rather than the example level, and is significantly cheaper than gradient-based methods.

#### Deliverable
For each analysed problem: a ranked list of influential training documents, categorised by source and similarity type. Aggregate analysis showing the training data profile behind each reasoning sub-skill.

### 6.5 Phase 5: Scale-Controlled Emergence Analysis

#### Objective
Using the controlled Instella family (1B → 3B → 3B-Math), characterise how reasoning capabilities emerge across scale and post-training.

#### Method

**Step 1 — Run all model variants on the full consistency suite:**

This mirrors Phase 3 but across all Instella variants, not just 3B.

**Step 2 — Classify capability transitions:**

For each reasoning sub-skill, categorise the 1B → 3B transition:

- **Scale-independent:** Both models show high reliability → basic capability not bottlenecked by scale
- **Emergent:** 3B shows high reliability, 1B shows near-zero → capability appears at scale
- **Amplified:** 3B shows higher reliability than 1B, but 1B shows partial capability → gradual improvement with scale
- **Scale-resistant:** Both models show low reliability → capability requires something beyond scale (e.g., different training data, different architecture)

**Step 3 — Chain-of-thought divergence analysis:**

For problems where 3B succeeds and 1B fails, align their chain-of-thought outputs step by step. Identify the exact step where 1B diverges:

- **Premature conclusion:** 1B skips reasoning steps
- **Error propagation:** 1B makes a small early mistake that cascades
- **Capacity failure:** 1B cannot track enough variables/constraints simultaneously
- **Strategy mismatch:** 1B attempts a fundamentally different (incorrect) approach

**Step 4 — Post-training effect analysis:**

Compare Instella-3B-Instruct and Instella-3B-Math to determine:

- Does RL post-training improve consistency (genuine reasoning improvement) or just accuracy (better pattern matching)?
- Does the improvement generalise beyond mathematics?
- Does RL obscure contamination signals (per arXiv:2510.02386)?

#### Deliverable
Capability transition matrix across model variants. Chain-of-thought divergence taxonomy. Analysis of post-training effects on reliability.

---

## 7. Models, Datasets, and Tools

### 7.1 Subject Models

| Model | HuggingFace ID | VRAM (FP16) | VRAM (4-bit) | T4 Feasibility |
|---|---|---|---|---|
| AMD OLMo-1B | `amd/AMD-OLMo-1B` | ~2 GB | ~1 GB | ✓ Full precision |
| Instella-3B | `amd/Instella-3B` | ~6 GB | ~3 GB | ✓ 4-bit quantised |
| Instella-3B-Instruct | `amd/Instella-3B-Instruct` | ~6 GB | ~3 GB | ✓ 4-bit quantised |
| Instella-3B-Math | `amd/Instella-3B-Math` | ~6 GB | ~3 GB | ✓ 4-bit quantised |
| Instella-3B-Long | `amd/Instella-3B-Long-Instruct` | ~6 GB | ~3 GB | ✓ 4-bit quantised |

All models run comfortably on a T4 (16 GB VRAM) with 4-bit quantisation via GPTQ or AWQ. The 1B model runs in full FP16 precision.

### 7.2 Reasoning Benchmarks

| Reasoning Sub-Skill | Benchmark | Test Size | Source |
|---|---|---|---|
| Arithmetic reasoning | GSM8K | 1,319 | Cobbe et al., 2021 |
| Mathematical reasoning | MATH (Level 1-3 subset) | ~2,500 | Hendrycks et al., 2021 |
| Logical deduction | LogiQA 2.0 | 1,600 | Liu et al., 2023 |
| Commonsense / science reasoning | ARC-Challenge | 1,172 | Clark et al., 2018 |
| Multi-step reasoning | BBH (selected subtasks) | ~1,500 | Suzgun et al., 2023 |
| Compositional reasoning | CLUTRR | ~500 | Sinha et al., 2019 |
| Logical reasoning (standardised tests) | ReClor | 500 | Yu et al., 2020 |
| Code reasoning | HumanEval | 164 | Chen et al., 2021 |
| Strategic reasoning | TTT-Bench | ~500 | Mishra et al., 2025 |

**Total benchmark problems: ~9,755**

We select BBH subtasks that target specific reasoning primitives: Boolean Expressions (logical), Causal Judgement (causal), Date Understanding (temporal), Logical Deduction (formal logic), Multi-Step Arithmetic, Navigate (spatial), Object Counting, Tracking Shuffled Objects (variable tracking).

### 7.3 Consistency Test Suites

For 800 problems sampled across benchmarks (stratified by benchmark, contamination level, and difficulty), we create 4 variants each:

| Perturbation Type | Applicable To | Description |
|---|---|---|
| Numerical perturbation | GSM8K, MATH, BBH arithmetic | Change values, preserve structure |
| Entity substitution | All benchmarks | Replace names, objects, contexts |
| Premise reordering | LogiQA, CLUTRR, BBH logic | Shuffle given information order |
| Irrelevant context injection | All benchmarks | Add distracting sentence |

**Total variant problems: ~3,200**
**Total problems including originals: ~12,955**

### 7.4 Training Data Sources

| Dataset | Role in Instella Training | Estimated Size | Index Strategy |
|---|---|---|---|
| DCLM | Stage 1 pretraining | ~3T tokens | Sample 5M passages |
| FineWeb-Edu | Stage 1 pretraining | ~1.3T tokens | Sample 3M passages |
| dolmino-mix-1124 | Stage 2 enrichment | ~100B tokens | Sample 2M passages |
| dm_math | Stage 2 enrichment | ~10B tokens | Full index |
| python-edu | Stage 2 enrichment | ~50B tokens | Sample 1M passages |
| Instella-GSM8K-synthetic | Stage 2 enrichment | ~50K problems | Full index |
| OpenMathInstruct-2 | Instella-Math SFT | 14M pairs | Sample 2M pairs |

**Total indexed passages: ~13-15 million**
**FAISS index size: ~6 GB (384-dim, flat) or ~2 GB (IVF+PQ)**

### 7.5 Utility Models and Tools

| Tool | Purpose | Size | Hardware |
|---|---|---|---|
| `all-MiniLM-L6-v2` | Sentence embedding | 80 MB | CPU |
| `gte-small` | Higher-quality embedding (alternative) | 67 MB | CPU |
| FAISS | Similarity search index | Library | CPU (GPU optional) |
| `transformers` + `bitsandbytes` | Model loading and quantisation | Library | T4 GPU |
| `vllm` or `transformers` | Inference engine | Library | T4 GPU |

No paid APIs. No external LLMs. The entire pipeline runs on a single T4 + CPU.

---

## 8. The Reasoning Reliability Atlas

The Reasoning Reliability Atlas is the central deliverable of this project — a diagnostic visualisation that maps each reasoning sub-skill across three dimensions:

### 8.1 Atlas Structure

For each reasoning sub-skill (arithmetic, logical deduction, constraint satisfaction, compositional reasoning, temporal reasoning, spatial reasoning, variable tracking, causal reasoning):

```
┌──────────────────────────────────────────────────────────────┐
│                    REASONING RELIABILITY ATLAS                │
├──────────────────┬───────────┬─────────────┬─────────────────┤
│ Sub-Skill        │ Accuracy  │ Consistency │ Reliability     │
│                  │           │             │ (Acc × Cons)    │
├──────────────────┼───────────┼─────────────┼─────────────────┤
│ Arithmetic       │ 0.78      │ 0.42        │ 0.33 ⚠ FRAGILE │
│  └ Contaminated  │ 0.92      │ 0.31        │ 0.29 ⚠         │
│  └ Clean         │ 0.61      │ 0.68        │ 0.41 ◐         │
├──────────────────┼───────────┼─────────────┼─────────────────┤
│ Logical Deduction│ 0.54      │ 0.71        │ 0.38 ◐ PARTIAL │
│  └ Contaminated  │ 0.62      │ 0.55        │ 0.34 ⚠         │
│  └ Clean         │ 0.49      │ 0.79        │ 0.39 ◐         │
├──────────────────┼───────────┼─────────────┼─────────────────┤
│ Constraint Sat.  │ 0.33      │ 0.82        │ 0.27 ✗ GAP     │
│  └ Clean         │ 0.33      │ 0.82        │ 0.27 ✗         │
├──────────────────┼───────────┼─────────────┼─────────────────┤
│ Compositional    │ 0.71      │ 0.73        │ 0.52 ✓ GENUINE │
│  └ Clean         │ 0.71      │ 0.73        │ 0.52 ✓         │
└──────────────────┴───────────┴─────────────┴─────────────────┘

Legend: ✓ Genuine (R > 0.50)  ◐ Partial (0.30 < R < 0.50)
        ⚠ Fragile (R < 0.30 with high accuracy)  ✗ Gap (low accuracy)
```

*(Note: All numbers above are illustrative. Actual values will be determined by experiments.)*

### 8.2 Attribution Layer

Each entry in the atlas is annotated with its training data profile:

```
Arithmetic (Clean, Reliability 0.41):
  Top training data sources:
    38% OpenWebMath (mathematical exposition)
    24% python-edu (implementation examples)
    19% FineWeb-Edu (word problems in educational contexts)
    12% dm_math (formal mathematics)
     7% Other
  
  → Attribution signature: DIVERSE, consistent with generalised learning
  → Recommendation: Increase diversity of multi-step arithmetic in training data;
     current data over-represents single-step calculations

Arithmetic (Contaminated, Reliability 0.29):
  Top training data sources:
    67% Instella-GSM8K-synthetic (near-duplicates)
    22% OpenMathInstruct-2 (close paraphrases)
    11% Other

  → Attribution signature: CONCENTRATED, consistent with memorisation
  → Recommendation: Deduplicate against benchmark test sets before training
```

### 8.3 Scale Comparison Layer

The atlas includes a comparison across model sizes:

```
                     AMD OLMo-1B    Instella-3B    Instella-3B-Math
                     Reliability    Reliability    Reliability
Arithmetic (Clean)      0.18           0.41           0.63
Logical Deduction       0.12           0.38           0.40
Constraint Sat.         0.05           0.27           0.28
Compositional           0.23           0.52           0.53

→ Observation: Math RL training improves mathematical reliability (+0.22)
   but not logical/constraint reasoning (+0.02/+0.01)
→ Implication: RL post-training teaches domain-specific reasoning, 
   not general reasoning transfer
```

*(All numbers illustrative.)*

---

## 9. Compute Budget and Feasibility

### 9.1 Detailed Compute Requirements

| Task | Hardware | Estimated Time | Memory |
|---|---|---|---|
| **Phase 1: Contamination Detection** | | | |
| Embed 13M training passages (MiniLM) | CPU (T4 host) | 18-24 hours | 8 GB RAM |
| Build FAISS IVF+PQ index | CPU | 2-3 hours | 16 GB RAM |
| Search 10K benchmark problems × top-20 | CPU | 15-30 minutes | 4 GB RAM |
| Compute n-gram overlap for top hits | CPU | 2-4 hours | 4 GB RAM |
| Manual validation (150 samples) | Human | 8-10 hours | N/A |
| **Phase 2: Baseline Evaluation** | | | |
| Instella-1B on ~10K problems | T4 GPU | 8-12 hours | 4 GB VRAM |
| Instella-3B (4-bit) on ~10K problems | T4 GPU | 20-30 hours | 6 GB VRAM |
| Instella-3B-Math (4-bit) on ~10K problems | T4 GPU | 20-30 hours | 6 GB VRAM |
| Instella-3B-Instruct (4-bit) on ~10K problems | T4 GPU | 20-30 hours | 6 GB VRAM |
| **Phase 3: Consistency Testing** | | | |
| Create consistency variants (scripted + manual QA) | CPU + Human | 2-3 weeks | N/A |
| Run 4 models × ~3,200 variants | T4 GPU | 4-6 days | 6 GB VRAM |
| **Phase 4: Attribution** | | | |
| Tier 1 — Embedding attribution (all problems) | CPU | Already done in Phase 1 | N/A |
| Tier 2 — TracIn-CP (500 problems × 100 candidates) | T4 GPU | 3-5 days | 14 GB VRAM |
| Tier 3 — Concept Influence (100 problems) | T4 GPU | 1-2 days | 10 GB VRAM |
| **Phase 5: Emergence Analysis** | | | |
| Cross-model comparison | CPU (analysis) | 2-3 days | N/A |
| CoT divergence analysis | CPU (analysis) | 1 week | N/A |

### 9.2 Total Compute Summary

| Resource | Total Required | Available |
|---|---|---|
| T4 GPU time | ~25-35 days | Continuous access assumed |
| CPU time | ~30-40 hours | Available on T4 host machine |
| RAM | 16 GB peak | Standard T4 instance |
| Disk | ~50 GB (embeddings + index + outputs) | Available |
| Human time (manual validation, variant QA) | ~4-5 weeks | PI's time |

This is well within the capability of a single T4 GPU. The compute-intensive phases (model inference) can run overnight. The FAISS indexing and search run entirely on CPU. No cloud GPU clusters, no paid APIs, no external services are required.

### 9.3 Cost Estimate

If using a cloud T4 instance (e.g., Google Cloud, ~$0.35/hour for preemptible):

```
35 days × 24 hours × $0.35 = ~$294
```

If using Google Colab Pro (~$10/month) with T4 access:

```
~5 months × $10 = $50 (with some session management overhead)
```

If using a university-provided GPU or personal hardware: **$0**.

---

## 10. Timeline and Milestones

### Month 1: Infrastructure and Contamination Analysis

**Weeks 1-2:**
- Download and preprocess Instella training data samples
- Set up embedding pipeline (MiniLM or GTE)
- Begin embedding training data passages
- Download all benchmark datasets

**Weeks 3-4:**
- Complete FAISS index construction
- Run contamination search for all benchmark problems
- Classify contamination levels (C/PC/N)
- Manual validation of 150 samples
- Write up contamination findings

**Milestone 1:** Contamination labels for all ~10K benchmark problems. Initial finding: "X% of GSM8K is contaminated through Instella-GSM8K-synthetic."

### Month 2: Baseline Evaluation and Consistency Suite Design

**Weeks 5-6:**
- Run all four Instella variants on all benchmarks
- Compute accuracy disaggregated by contamination level
- Begin consistency variant creation (scripted perturbations)

**Weeks 7-8:**
- Complete consistency variant creation
- Quality-check all variants
- Begin running consistency experiments
- Initial accuracy-by-contamination analysis

**Milestone 2:** Accuracy tables showing the contamination gap. Initial finding: "Instella-3B scores X% on contaminated GSM8K problems but Y% on clean ones."

### Month 3: Consistency Testing and Attribution

**Weeks 9-10:**
- Complete consistency experiments across all models
- Compute Reliability Scores for all tested problems
- Cross-reference reliability with contamination levels
- Begin embedding-based attribution analysis

**Weeks 11-12:**
- Run TracIn-CP attribution on 500 selected problems
- Run Concept Influence on 100 validation problems
- Build the Reasoning Reliability Atlas
- Scale-controlled emergence analysis

**Milestone 3:** Complete Reasoning Reliability Atlas with attribution layer. Key finding: the training data profile behind each reasoning sub-skill.

### Month 4: Analysis, Data Curation Guidelines, and Paper Writing

**Weeks 13-14:**
- Complete emergence analysis (1B vs 3B vs 3B-Math)
- Formulate data curation recommendations
- Begin paper writing (Introduction, Related Work, Methodology)

**Weeks 15-16:**
- Complete paper (Results, Analysis, Discussion, Conclusion)
- Create figures and tables
- Internal review and revision
- Share draft with Dr. Liu for feedback

**Milestone 4:** Complete paper draft ready for venue submission.

### Month 5: Revision and Submission

**Weeks 17-18:**
- Incorporate feedback from Dr. Liu
- Run any additional experiments identified during review
- Polish figures, tables, and writing

**Weeks 19-20:**
- Final revision
- Prepare supplementary materials
- Submit to target venue

**Milestone 5:** Paper submitted.

---

## 11. Paper Structure

```
Title: Reasoning or Remembering? Diagnosing and Attributing
       Reasoning Capabilities in Fully Open Language Models

Authors: Govind Arun Kumar¹, Jiang Liu²
         ¹ University of Maryland, College Park
         ² AMD GenAI

Abstract: (~250 words)
  First study combining contamination analysis, consistency-based 
  reliability evaluation, and training data attribution on a fully 
  open LLM family. We introduce the Reasoning Reliability Atlas — 
  a diagnostic framework that distinguishes genuine reasoning from 
  memorisation by measuring accuracy × consistency across semantically 
  equivalent problem variants, then tracing reliable (and unreliable) 
  reasoning back to specific training data sources. Applied to AMD's 
  Instella family (1B–3B), our findings reveal...

1. Introduction (1.5 pages)
   - The accuracy illusion in reasoning evaluation
   - Why full openness enables new science
   - Summary of contributions

2. Related Work (2 pages)
   - Reasoning benchmarks and robustness evaluation
   - Training data contamination detection
   - Training data attribution methods
   - Connection to our unified approach

3. Methodology (3 pages)
   3.1 Contamination Detection via Training Data Search
       - Embedding-based near-duplicate detection
       - N-gram overlap verification
       - Contamination classification scheme
   3.2 Consistency-Based Reliability Evaluation
       - Consistency cluster design
       - Perturbation taxonomy
       - The Reliability Score metric
   3.3 Training Data Attribution
       - Embedding-based attribution
       - TracIn-CP for gradient-based attribution
       - Concept Influence validation
   3.4 Scale-Controlled Emergence Analysis

4. Experimental Setup (1.5 pages)
   4.1 Models: AMD Instella family
   4.2 Benchmarks: 9 benchmarks, 8 reasoning sub-skills
   4.3 Consistency suites: 3,200 variant problems
   4.4 Training data: 13M indexed passages from Instella's pretraining

5. Results (4 pages)
   5.1 How Much "Reasoning" Is Memorisation?
       - Contamination rates by benchmark
       - Accuracy gap: contaminated vs clean
   5.2 The Reasoning Reliability Atlas
       - Reliability profiles across 8 sub-skills
       - Contamination × reliability interaction
       - The fragile pattern-matching signature
   5.3 What Training Data Produces Reliable Reasoning?
       - Attribution profiles by sub-skill
       - Concentrated vs. diverse attribution signatures
       - Training data source analysis
   5.4 Reasoning Across Scale and Post-Training
       - 1B → 3B capability transitions
       - Effect of math-specific RL on reliability
       - Chain-of-thought divergence taxonomy

6. Data Curation Guidelines for Reasoning (1 page)
   - Per-sub-skill recommendations
   - Training data density × diversity correlation
   - Decontamination recommendations

7. Discussion and Limitations (1 page)
   - Threats to validity
   - Generalisation beyond Instella
   - Compute constraints and approximation quality

8. Conclusion (0.5 pages)

References

Appendix:
   A. Full contamination results
   B. Consistency suite examples
   C. Attribution case studies
   D. Detailed per-benchmark results
```

**Estimated length:** 9-10 pages main text + appendix (standard NeurIPS/ICLR format)

---

## 12. Expected Contributions

### 12.1 Methodological Contributions

1. **The Reliability Score:** A novel evaluation metric that captures both accuracy and consistency, providing a more faithful measure of reasoning capability than accuracy alone. This metric is applicable to any LLM, not just Instella.

2. **Contamination-aware reasoning evaluation framework:** A systematic methodology for evaluating reasoning in the presence of (known) training data contamination. The framework separates genuine capability from memorisation.

3. **Multi-tier attribution pipeline:** A compute-efficient combination of embedding-based search, TracIn-CP, and Concept Influence that works on a single consumer GPU — making training data attribution accessible to the broader research community.

### 12.2 Empirical Contributions

4. **First contamination audit of a fully open reasoning model:** Comprehensive contamination analysis across 9 benchmarks on Instella, with ground-truth training data access rather than indirect detection methods.

5. **The Reasoning Reliability Atlas:** A diagnostic map of reasoning capabilities in the Instella family, characterising which sub-skills are genuine, fragile, or absent — with explanatory attribution.

6. **Controlled emergence study:** The first emergence analysis on same-pipeline models with both base and reasoning-specialised variants, isolating scale from data from post-training effects.

### 12.3 Practical Contributions

7. **Data curation guidelines:** Actionable recommendations for improving reasoning in future Instella versions (and, by generalisation, other LLMs) through targeted training data curation.

8. **Open tooling:** The contamination search pipeline, consistency test generation scripts, and reliability scoring code will be released as an open-source toolkit for the community.

### 12.4 Why This Matters Beyond Instella

While our experiments focus on Instella, every contribution generalises:

- The Reliability Score can be applied to any model.
- The consistency test suites can be used to evaluate any LLM.
- The methodological framework (contamination → reliability → attribution) can be replicated on any model family that releases training data. As more model families move toward openness, this framework becomes increasingly relevant.
- The findings about which training data produces genuine reasoning inform the entire field's data curation strategies.

---

## 13. Risk Analysis and Contingency Plans

### Risk 1: Instella's training data is less accessible than documented

**Probability:** Low. AMD has released data source names and recipes. Some large datasets (DCLM) may require downloading terabytes.

**Mitigation:** Start with the most accessible datasets (Instella-GSM8K-synthetic, dm_math, python-edu, OpenMathInstruct-2) where contamination is most likely. Use metadata and sampling for larger corpora.

### Risk 2: TracIn-CP is too expensive on T4 for Instella-3B

**Probability:** Medium. Gradient computation for 3B parameters in 4-bit quantisation may exceed T4 memory during backpropagation.

**Mitigation:** (a) Use gradient checkpointing to trade compute for memory. (b) Compute TracIn on Instella-1B instead (1B fits in FP16). (c) Fall back to Concept Influence (Kowal et al., 2026), which requires only forward passes. (d) Use embedding-based attribution as the primary method and frame gradient-based attribution as validation.

### Risk 3: Contamination levels are too low to produce interesting findings

**Probability:** Low, given the presence of Instella-GSM8K-synthetic and OpenMathInstruct-2 in the training pipeline. But possible for non-mathematical benchmarks.

**Mitigation:** If contamination rates are low (<5%) across all benchmarks, this itself is a positive finding for Instella. Reframe the paper as "Instella's reasoning is mostly genuine" with the Reliability Atlas as the primary contribution.

### Risk 4: Consistency test suites contain bugs or biases

**Probability:** Medium. Perturbation generation can introduce unintended difficulty changes.

**Mitigation:** Manual quality check of all variants (sampling 20% of all created problems). Ensure that perturbations preserve problem difficulty by consulting the GSM-Symbolic methodology. Report inter-annotator agreement on difficulty preservation.

### Risk 5: RL post-training obscures contamination signals (per arXiv:2510.02386)

**Probability:** High for Instella-Math specifically.

**Mitigation:** Run contamination analysis on Instella-3B-Instruct (pre-RL) as well as Instella-3B-Math (post-RL). If contamination signals differ, this is itself an interesting finding about how RL affects memorisation.

### Risk 6: Someone publishes a similar study before submission

**Probability:** Low-Medium. The specific combination (contamination + consistency + attribution on Instella) is narrow enough to be unlikely, but the general topic (reasoning vs memorisation) is hot.

**Mitigation:** Move quickly. The 4-month timeline targets ICLR 2027 (likely October 2026 deadline). If scooped on Instella specifically, extend to other open models (OLMo 2, Pythia) for a broader comparative study.

---

## 14. Target Venues and Deadlines

| Venue | Typical Deadline | Expected Notification | Notes |
|---|---|---|---|
| **ICLR 2027** | ~October 2026 | ~January 2027 | Primary target. Strong fit for evaluation + analysis papers. |
| **NeurIPS 2027** | ~May 2027 | ~September 2027 | Backup. Broader audience. |
| **COLM 2027** | ~TBD (likely March 2027) | ~TBD | Ideal venue — Conference on Language Modeling. |
| **ACL 2027** | ~January 2027 | ~May 2027 | Good fit for NLP-focused framing. |
| **EMNLP 2027** | ~June 2027 | ~October 2027 | Strong fit for benchmark/evaluation papers. |

**Recommended strategy:** Target ICLR 2027 as primary (highest prestige, strong evaluation track). If timeline slips, ACL 2027 or COLM 2027 as alternatives.

The Datasets and Benchmarks track at NeurIPS and ICLR typically has higher acceptance rates and is well-suited to this type of contribution.

---

## 15. Complete Reference List

### Reasoning Benchmarks

1. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.

2. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., Steinhardt, J. (2021). Measuring Mathematical Problem Solving with the MATH Dataset. *NeurIPS 2021 Datasets & Benchmarks*. arXiv:2103.03874.

3. Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., Tafjord, O. (2018). Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. *arXiv:1803.05457*.

4. Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., Zhang, Y. (2020). LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning. *IJCAI 2020*. arXiv:2007.08124.

5. Liu, H., et al. (2023). LogiQA 2.0. *IEEE/ACM TASLP 2023*. DOI: 10.1109/TASLP.2023.3293046.

6. Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H.W., Chowdhery, A., Le, Q.V., Chi, E.H., Zhou, D., Wei, J. (2023). Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. *ACL Findings 2023*. arXiv:2210.09261.

7. Srivastava, A., Rastogi, A., Rao, A., et al. (2023). Beyond the Imitation Game (BIG-Bench). *TMLR 2023*. arXiv:2206.04615.

8. Sinha, K., Sodhani, S., Dong, J., Pineau, J., Hamilton, W.L. (2019). CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text. *EMNLP-IJCNLP 2019*. arXiv:1908.06177.

9. Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. *arXiv:2107.03374*.

10. Yu, W., Jiang, Z., Dong, Y., Feng, J. (2020). ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning. *ICLR 2020*. arXiv:2002.04326.

11. Mishra, P., Liu, J., Wu, J., Yu, X., Liu, Z., Barsoum, E. (2025). TTT-Bench: A Benchmark for Evaluating Reasoning Ability with Simple and Novel Tic-Tac-Toe-style Games. *EMNLP 2025*. arXiv:2506.10209.

12. Sileo, D., Lernould, A. (2023). MindGames: Targeting Theory of Mind in LLMs with Dynamic Epistemic Modal Logic. *EMNLP Findings 2023*. arXiv:2305.03353.

13. Kazemi, S.M., et al. (2025). BIG-Bench Extra Hard (BBEH). *arXiv:2502.19187*.

14. Zeng, Z., et al. (2023). MR-GSM8K: A Meta-Reasoning Benchmark for LLM Evaluation. *arXiv:2312.17080*.

### Reasoning Robustness and Consistency

15. Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S., Farajtabar, M. (2024). GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. *ICLR 2025*. arXiv:2410.05229.

16. Li, Q., Cui, L., Zhao, X., Kong, L., Bi, W. (2024). GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers. *ACL 2024*. arXiv:2402.19255.

17. Huang, K., et al. (2025). MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations. *arXiv:2502.06453*.

18. Srivastava, S., et al. (2024). Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap. *arXiv:2402.19450*.

19. Gulati, A., Miranda, B., Chen, E., Xia, E., Fronsdal, K., Dumont, B., Obbad, E., Koyejo, S. (2025). Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs. *arXiv:2508.08292*.

20. McCoy, R.T., Yao, S., Friedman, D., Hardy, M., Griffiths, T.L. (2023). Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve. *arXiv:2309.13638*. PNAS 2024.

21. Dziri, N., Lu, X., Sclar, M., et al. (2023). Faith and Fate: Limits of Transformers on Compositionality. *NeurIPS 2023 Spotlight*. arXiv:2305.18654.

22. Yu, X., Zhou, B., Cheng, H., Roth, D. (2024). ReasonAgain. *arXiv:2410.19056*.

23. Path-of-Thoughts. (2024). *arXiv:2412.17963*.

### Training Data Contamination

24. Brown, T.B., et al. (2020). Language Models are Few-Shot Learners (GPT-3). *NeurIPS 2020*. arXiv:2005.14165.

25. Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramèr, F., Zhang, C. (2023). Quantifying Memorization Across Neural Language Models. *ICLR 2023*. arXiv:2202.07646.

26. Carlini, N., Tramèr, F., Wallace, E., et al. (2021). Extracting Training Data from Large Language Models. *USENIX Security 2021*. arXiv:2012.07805.

27. Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., Zettlemoyer, L. (2024). Detecting Pretraining Data from LLMs (Min-K% Prob). *ICLR 2024*. arXiv:2310.16789.

28. Yang, S., Chiang, W.-L., Zheng, L., Gonzalez, J.E., Stoica, I. (2023). Rethinking Benchmark and Contamination for Language Models with Rephrased Samples. *arXiv:2311.04850*.

29. Zhou, K., et al. (2023). Don't Make Your LLM an Evaluation Benchmark Cheater. *arXiv:2311.01964*.

30. Sainz, O., et al. (2023). NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark. *EMNLP Findings 2023*.

31. Fu, Y., et al. (2024). A Survey on Benchmark Data Contamination for LLMs. *arXiv:2406.04244*.

32. On the Fragility of Benchmark Contamination Detection in Reasoning Models. (2025). *arXiv:2510.02386*.

33. Detecting Data Contamination in LLMs via In-Context Learning. (2025). *arXiv:2510.27055*.

### Training Data Attribution

34. Koh, P.W., Liang, P. (2017). Understanding Black-box Predictions via Influence Functions. *ICML 2017 Best Paper*. arXiv:1703.04730.

35. Pruthi, G., Liu, F., Sundararajan, M., Kale, S. (2020). Estimating Training Data Influence by Tracing Gradient Descent (TracIn). *NeurIPS 2020*. arXiv:2002.08484.

36. Ilyas, A., Park, S.M., Engstrom, L., Leclerc, G., Mądry, A. (2022). Datamodels: Predicting Predictions from Training Data. *ICML 2022*. arXiv:2202.00622.

37. Park, S.M., Georgiev, K., Ilyas, A., Leclerc, G., Mądry, A. (2023). TRAK: Attributing Model Behavior at Scale. *ICML 2023*. arXiv:2303.14186.

38. Grosse, R., Bae, J., Anil, C., et al. (2023). Studying Large Language Model Generalization with Influence Functions. *arXiv:2308.03296*. (Anthropic).

39. Chang, T.A., et al. (2024). TrackStar: Scalable Influence and Fact Tracing for LLM Pretraining. *arXiv:2410.17413*.

40. Wu, K., Pang, L., Shen, H., Cheng, X. (2024). Enhancing Training Data Attribution for LLMs with Fitting Error Consideration (DDA). *arXiv:2410.01285*.

41. Kowal, M., Paulo, G., Jaburi, L., et al. (2026). Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution. *arXiv:2602.14869*.

42. Bae, J., Ng, N., Lo, A., Ghassemi, M., Grosse, R. (2022). If Influence Functions are the Answer, Then What is the Question? *arXiv:2209.05364*.

43. Do Influence Functions Work on Large Language Models? (2024). *arXiv:2409.19998*.

44. Yeh, C.-K., et al. (2022). First is Better Than Last for Language Data Influence. *NeurIPS 2022*. arXiv:2202.11844.

### Emergence and Scaling

45. Wei, J., Tay, Y., Bommasani, R., et al. (2022). Emergent Abilities of Large Language Models. *TMLR 2022*. arXiv:2206.07682.

46. Schaeffer, R., Miranda, B., Koyejo, S. (2023). Are Emergent Abilities of Large Language Models a Mirage? *NeurIPS 2023 Outstanding Paper*. arXiv:2304.15004.

47. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. arXiv:2201.11903.

48. Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.

### Instella and Related Open Models

49. Liu, J., Wu, J., Yu, X., Su, Y., Mishra, P., Ramesh, G., Ranjan, S., Manem, C., Sun, X., Wang, Z., Brahma, P.P., Liu, Z., Barsoum, E. (2025). Instella: Fully Open Language Models with Stellar Performance. *arXiv:2511.10628*.

50. Groeneveld, D., Beltagy, I., Walsh, P., et al. (2024). OLMo: Accelerating the Science of Language Models. *ACL 2024*. arXiv:2402.00838.

51. Walsh, P., Soldaini, L., Groeneveld, D., et al. (2025). OLMo 2. *arXiv:2501.00656*.

52. Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. *ICML 2023*. arXiv:2304.01373.

### Data-Centric AI and Training Data

53. Lewkowycz, A., Andreassen, A., Dohan, D., et al. (2022). Minerva: Solving Quantitative Reasoning Problems with Language Models. *NeurIPS 2022*. arXiv:2206.14858.

54. Azerbayev, Z., Schoelkopf, H., Paster, K., et al. (2024). Llemma: An Open Language Model for Mathematics. *ICLR 2024*. arXiv:2310.10631.

55. Paster, K., Dos Santos, M., Azerbayev, Z., Ba, J. (2024). OpenWebMath. *ICLR 2024*. arXiv:2310.06786.

56. Toshniwal, S., et al. (2024). OpenMathInstruct-2. *arXiv:2410.01560*.

57. Li, J., Fang, A., Smyrnis, G., et al. (2024). DataComp-LM (DCLM). *NeurIPS 2024 D&B*. arXiv:2406.11794.

58. Penedo, G., et al. (2024). FineWeb / FineWeb-Edu. *NeurIPS 2024 D&B*. arXiv:2406.17557.

59. Yue, X., Zheng, T., Zhang, G., Chen, W. (2024). MAmmoTH2: Scaling Instructions from the Web. *NeurIPS 2024*. arXiv:2405.03548.

60. Liu, B., et al. (2023). TinyGSM. *arXiv:2312.09241*.

61. Ben Allal, L., et al. (2025). SmolLM2 — Data-Centric Training of a Small LM. *arXiv:2502.02737*.

62. Kocetkov, D., et al. (2022). The Stack. *arXiv:2211.15533*.

### Embedding and Retrieval Infrastructure

63. Johnson, J., Douze, M., Jégou, H. (2019). Billion-scale similarity search with GPUs (FAISS). *IEEE TBD, vol. 7 no. 3*. arXiv:1702.08734.

64. Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E., Lomeli, M., Hosseini, L., Jégou, H. (2024). The Faiss library. *arXiv:2401.08281*.

65. Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP-IJCNLP 2019*. arXiv:1908.10084.

66. Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *NeurIPS 2020*. arXiv:2002.10957.

67. Li, Z., Zhang, X., Zhang, Y., Long, D., Xie, P., Zhang, M. (2023). GTE — General Text Embeddings. *arXiv:2308.03281*.

68. Malkov, Y., Yashunin, D.A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (HNSW). *TPAMI 2018*. arXiv:1603.09320.

69. Jégou, H., Douze, M., Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. *IEEE TPAMI 2011*.

### Ivanova's Statistical Critique

70. Ivanova, D.R. (2024). On Some (Fixable) Limitations of "Understanding the Limitations of Mathematical Reasoning in LLMs." Blog post, 22 Oct 2024. Available at: desirivanova.com/post/gsm-symbolic/.

### Jiang Liu's Relevant Work (Proposed Collaborator)

71. Liu, J., Wu, J., Yu, X., et al. (2025). Instella: Fully Open Language Models with Stellar Performance. *arXiv:2511.10628*.

72. Mishra, P., Liu, J., Wu, J., Yu, X., Liu, Z., Barsoum, E. (2025). TTT-Bench: A Benchmark for Evaluating Reasoning Ability with Simple and Novel Tic-Tac-Toe-style Games. *EMNLP 2025*. arXiv:2506.10209.

73. Huang, C., Zhang, Z., Liu, J., Sun, X., Wu, J., Yu, X., Wang, Z., Xu, C., Barsoum, E., Liu, Z. (2026). DRIFT: Directional Reasoning Injection for Fine-Tuning MLLMs. *ACL Findings 2026*.

74. Xu, Z., Yu, X., Zhou, B., Liu, J., Wu, J., Wang, Z., Sun, X., Chen, H., Liu, Z. (2026). RULES: Reliable Use of Lemmas via Eligibility Reasoning and Section-Aware Reinforcement Learning. *ACL 2026*.

75. Wang, H., Yu, X., Wu, J., Liu, J., Sun, X., Bansal, M., Liu, Z. (2026). Stabilizing Efficient Reasoning with Step-Level Advantage Selection (SAS). *ACL 2026*.

76. Zhou, Y., Li, J., Su, Y., Ramesh, G., Zhu, Z., Long, X., Zhao, C., Pan, J., Yu, X., Wang, Z., Du, K., Wu, J., Sun, X., Liu, J., Yu, Q., Chen, H., Liu, Z., Barsoum, E. (2025). APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation. *arXiv 2025*.

77. Wang, X., Liu, J., Huang, C., Yu, X., Wang, Z., Sun, X., Wu, J., Yuille, A., Barsoum, E., Liu, Z. (2026). XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models. *ICLR 2026*.

78. Guo, Y., Liu, J., Wang, Z., Chen, H., Sun, X., Zhao, Y., Wu, J., Yu, X., Liu, Z., Barsoum, E. (2026). ImageDoctor: Diagnosing Text-to-Image Generation via Grounded Image Reasoning. *ICLR 2026*.

79. Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., Liu, Z. (2026). Latent Visual Reasoning. *ICLR 2026*.

80. Schmidgall, S., Su, Y., Wang, Z., Sun, X., Wu, J., Yu, X., Liu, J., Moor, M., Liu, Z., Barsoum, E. (2025). Agent Laboratory: Using LLM Agents as Research Assistants. *EMNLP Findings 2025*.

81. Lin, J., Wu, J., Sun, X., Wang, Z., Liu, J., Su, Y., Yu, X., Chen, H., Luo, J., Liu, Z., Barsoum, E. (2025). VideoMarathon: Unleashing Hour-Scale Video Training for Long Video-Language Understanding. *NeurIPS 2025 Spotlight*.

---

## Appendix A: Key Methodological Decisions

### A.1 Why Embedding Similarity Over N-gram Overlap Alone

N-gram overlap (following Brown et al., 2020) catches exact and near-exact duplicates but misses paraphrased contamination entirely (Yang et al., 2023). Embedding similarity captures semantic overlap regardless of surface form. We use both: embedding similarity as the primary detection method, n-gram overlap as a high-precision validator.

### A.2 Why Reliability Score Over Accuracy Alone

Accuracy tells you how often the model is right. Consistency tells you whether the model is right for the right reasons. The product (Reliability = Accuracy × Consistency) captures both dimensions in a single number.

The key insight is that **pattern matching produces high accuracy but low consistency**: the model recognises a specific problem formulation but fails on equivalent formulations. **Genuine reasoning produces correlated accuracy and consistency**: the model either understands the problem type (and succeeds consistently) or does not (and fails consistently).

### A.3 Why TracIn-CP Over Full Influence Functions

Full influence functions require inverse Hessian computation, which is infeasible for 3B parameters on a T4. TracIn-CP requires only gradient dot products at saved checkpoints, which are feasible with gradient checkpointing and FAISS pre-filtering.

The key tradeoff: TracIn-CP provides a first-order approximation that is noisier than full influence functions but correlates well with ground truth for identifying the most influential training examples (Pruthi et al., 2020).

### A.4 Why 4-bit Quantisation Is Acceptable

Our study focuses on behaviour (what the model gets right/wrong and how consistently) rather than representation (what the model represents internally). 4-bit quantisation preserves task accuracy within 1-2% of full precision for most benchmarks. We validate this assumption by running a subset of experiments on Instella-1B in full precision and confirming that accuracy and consistency rankings are preserved.

### A.5 Handling the GSM-Symbolic Controversy

Mirzadeh et al.'s claim that LLMs "replicate reasoning steps from their training data" is treated as a hypothesis to test, not a settled finding. Following Ivanova's critique (2024), we:

1. Report statistical significance for all accuracy comparisons
2. Use multiple perturbation types (not just irrelevant clause injection)
3. Control for difficulty changes introduced by perturbations
4. Report variance across multiple runs (temperature sampling) where feasible

Our contamination analysis provides what GSM-Symbolic could not: direct evidence of whether near-duplicate training data exists for each problem. This transforms the debate from indirect inference ("performance dropped, therefore memorisation") to direct evidence ("this problem has/does not have a near-duplicate in training data, and performance did/did not drop on variants").

---

## Appendix B: Instella Model Access and Resources

| Resource | URL |
|---|---|
| Instella-3B weights | https://huggingface.co/amd/Instella-3B |
| Instella-3B-Instruct weights | https://huggingface.co/amd/Instella-3B-Instruct |
| Instella-3B-Math weights | https://huggingface.co/amd/Instella-3B-Math |
| Instella-3B-Long-Instruct weights | https://huggingface.co/amd/Instella-3B-Long-Instruct |
| Instella training code | https://github.com/AMD-AGI/Instella |
| Instella-GSM8K-synthetic dataset | https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic |
| Instella arXiv report | https://arxiv.org/abs/2511.10628 |
| Instella launch blog | https://rocm.blogs.amd.com/artificial-intelligence/instella/README.html |
| Instella-Math blog | https://rocm.blogs.amd.com/artificial-intelligence/instella-math-language/README.html |
| Instella-Long blog | https://rocm.blogs.amd.com/artificial-intelligence/instella-long-context/README.html |
| FAISS library | https://github.com/facebookresearch/faiss |
| Sentence-Transformers | https://www.sbert.net/ |

---

## Appendix C: Glossary

- **Contamination:** The presence of evaluation benchmark items (or close variants) in a model's training data.
- **Consistency:** The degree to which a model produces the same outcome (correct or incorrect) across semantically equivalent formulations of a problem.
- **Reliability Score:** Accuracy × Consistency. A measure of genuine reasoning capability.
- **Attribution:** Tracing a model's output back to the training data that most influenced it.
- **Reasoning sub-skill:** A specific component of reasoning (e.g., arithmetic, logical deduction, constraint satisfaction) that can be tested independently.
- **Reasoning Reliability Atlas:** A diagnostic visualisation mapping each reasoning sub-skill's accuracy, consistency, reliability, and training data profile.
- **TracIn-CP:** A training data attribution method based on gradient dot products at saved checkpoints.
- **Concept Influence:** A recent attribution method using interpretability probes that is cheaper than gradient-based methods.
- **FAISS:** Facebook AI Similarity Search — a library for efficient similarity search over dense vectors.
- **Fully open model:** A model where weights, training data, training code, and data recipes are all publicly available.

---

*Document prepared: May 2026*
*Last updated: May 24, 2026*
*Contact: govind02@umd.edu / govind123.ga@gmail.com*
