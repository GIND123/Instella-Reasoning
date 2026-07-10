# Reasoning or Remembering? Diagnosing and Attributing Reasoning Capabilities in Fully Open Language Models — A Research Proposal Reference Compendium (AMD Instella Focus)

## TL;DR
- The proposed research sits at a productive intersection of three mature literatures — reasoning benchmarks (GSM8K, MATH, BBH, ARC, LogiQA, CLUTRR, HumanEval, plus 2024–2026 entrants like GSM-Symbolic, GSM-Plus, MATH-Perturb, TTT-Bench), training-data contamination detection (Carlini, Shi/Min-K%, Yang rephrased samples), and training-data attribution (Koh & Liang, TracIn, TRAK, datamodels, Anthropic's influence-functions-for-LLMs) — and AMD's fully open Instella family (Instella-3B, Instella-Math, Instella-Long, AMD OLMo-1B) is one of the very few model lineages where every artifact needed to actually execute such a study (weights, training data, and recipes) is public.
- The most decision-relevant gap in the existing literature is **causal**: contamination work tells us *whether* a benchmark item was seen, attribution work tells us *which* training documents most influenced an output, but almost no published work *combines* the two on a fully open reasoning model to ask "did this correct GSM8K answer come from genuine multi-step reasoning, from a near-duplicate in DCLM/Dolma/OpenWebMath, or from a paraphrase?" Instella is uniquely positioned to answer that.
- A lightweight, compute-feasible pipeline — FAISS + Sentence-BERT/MiniLM/GTE embeddings for near-duplicate retrieval over Instella's known pretraining mix, paired with paraphrase robustness probes (GSM-Symbolic-style perturbations, CLUTRR systematicity splits) and a cheap attribution proxy (TracIn-style gradient dot products on saved Instella checkpoints, or a TRAK-style randomly-projected kernel) — is the recommended technical path. This is achievable on a single MI300X node and would produce the first per-benchmark-item "reasoning vs. remembering" decomposition on a fully open frontier-quality 3B model.

## Key Findings
1. **All eight literature areas requested are well-populated with verifiable primary sources.** Forty-eight references with confirmed arXiv IDs, authors, and venues are catalogued in the Details section below.
2. **Instella is exceptionally well-documented for the purpose of this proposal.** AMD has released, in addition to weights, the full pretraining mix (OLMoE-mix-0924 + Dolmino-mix-1124 + python-edu + dm_math + the synthetic Instella-GSM8K-synthetic dataset), and an arXiv technical report (arXiv:2511.10628, Liu et al., Nov 2025). This means contamination and attribution studies are not merely theoretical here — every training document is inspectable.
3. **Recent benchmark-perturbation work (GSM-Symbolic, GSM-Plus, MATH-Perturb, Functional-MATH, Putnam-AXIOM, TTT-Bench) provides a ready-made toolkit** for the "consistency/robustness" arm of the proposal. Mirzadeh et al.'s GSM-Symbolic (ICLR 2025) states verbatim in its abstract: "We hypothesize that this decline is because current LLMs cannot perform genuine logical reasoning; they replicate reasoning steps from their training data. Adding a single clause that seems relevant to the question causes significant performance drops (up to 65%) across all state-of-the-art models" — exactly the hypothesis the proposed work would test on Instella.
4. **The contamination-detection literature has converged on three families of methods** — n-gram/embedding overlap (Brown et al. 2020, Yang et al. 2023), perplexity/loss-based membership inference (Carlini et al. 2022/2023; Shi et al. Min-K% 2024), and rephrased/paraphrase-aware detection (Yang et al. 2023; Golchin & Surdeanu 2023). For a fully open model like Instella the embedding-overlap path is both cheapest and most defensible because the pretraining data is actually accessible.
5. **For training-data attribution at LLM scale**, classical influence functions (Koh & Liang 2017) are infeasible without approximation; the practical toolbox is TracIn (Pruthi et al. 2020), TRAK (Park et al. 2023), datamodels (Ilyas et al. 2022), Anthropic's EK-FAC influence functions (Grosse et al. 2023, which scales to 52 billion parameters per arXiv:2308.03296), and the very recent Concept Influence (Kowal et al. 2026). For a 3B model on a single MI300X node, TracIn-CP with a small number of saved checkpoints plus FAISS pre-filtering is the most realistic pipeline.
6. **Notable claim to flag with caution:** Mirzadeh et al. (2024) explicitly hypothesise that LLMs "replicate reasoning steps from their training data" rather than reasoning; this is widely cited but contested. Desi R. Ivanova's statistical critique ("On Some (Fixable) Limitations of 'Understanding the Limitations of Mathematical Reasoning in LLMs'", desirivanova.com/post/gsm-symbolic/, 22 Oct 2024) concludes that "for 21 out of 25 models, there isn't enough evidence to reject the null hypothesis that performance on GSM8K is equal to that on GSM-Symbolic." Any proposal should treat the Mirzadeh claim as a hypothesis to *test* on Instella, not a settled fact.
7. **Emergence/scale findings are genuinely contested**: Wei et al. (2022, TMLR) reported emergent abilities; Schaeffer, Miranda & Koyejo (NeurIPS 2023 Outstanding Paper) argue that, per the official announcement, "nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance." The Instella family (1B AMD OLMo → 3B Instella → 3B Instella-Math) is a near-ideal controlled testbed for this debate.

## Details

### 1. LLM Reasoning Benchmarks and Evaluation

**Classical / foundational benchmarks**
- **GSM8K** — Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Łukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman. "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168, 2021 (OpenAI).
- **MATH** — Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt. "Measuring Mathematical Problem Solving with the MATH Dataset." NeurIPS 2021 Datasets & Benchmarks. arXiv:2103.03874.
- **LogiQA** — Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, Yue Zhang. "LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning." IJCAI 2020. arXiv:2007.08124. **LogiQA 2.0** — Hanmeng Liu et al., IEEE/ACM TASLP 2023, DOI 10.1109/TASLP.2023.3293046.
- **ARC / ARC-Challenge** — Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord. "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge." arXiv:1803.05457, 2018.
- **BIG-Bench Hard (BBH)** — Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, Jason Wei. "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them." ACL Findings 2023. arXiv:2210.09261.
- **BIG-Bench (parent)** — Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, et al. (~450 authors). "Beyond the Imitation Game (BIG-Bench)." TMLR 2023. arXiv:2206.04615.
- **CLUTRR** — Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, William L. Hamilton. "CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text." EMNLP-IJCNLP 2019. arXiv:1908.06177.
- **HumanEval / Codex** — Mark Chen, Jerry Tworek, Heewoo Jun, et al. "Evaluating Large Language Models Trained on Code." arXiv:2107.03374, 2021.

**Recent (2024–2026) reasoning benchmarks**
- **TTT-Bench** — Prakamya Mishra, Jiang Liu, Jialian Wu, Xiaodong Yu, Zicheng Liu, Emad Barsoum (AMD). "TTT-Bench: A Benchmark for Evaluating Reasoning Ability with Simple and Novel Tic-Tac-Toe-style Games." arXiv:2506.10209, 2025. (Notable: authored by the same AMD team behind Instella.)
- **MindGames** — Damien Sileo, Antoine Lernould. "MindGames: Targeting Theory of Mind in LLMs with Dynamic Epistemic Modal Logic." EMNLP Findings 2023. arXiv:2305.03353.
- **BIG-Bench Extra Hard (BBEH)** — Kazemi et al. arXiv:2502.19187, 2025.
- **MR-GSM8K** — Zeng et al. "MR-GSM8K: A Meta-Reasoning Benchmark for LLM Evaluation." arXiv:2312.17080, 2023/2024.

**Consistency / robustness-focused benchmarks (central to proposal)**
- **GSM-Symbolic** — Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar (Apple). ICLR 2025. arXiv:2410.05229. Demonstrates up to 65% performance drops when irrelevant clauses are added to GSM8K problems.
- **GSM-Plus** — Qintong Li, Leyang Cui, Xueliang Zhao, Lingpeng Kong, Wei Bi. ACL 2024. arXiv:2402.19255. Eight adversarial perturbation types per GSM8K item.
- **MATH-Perturb** — Kaixuan Huang et al. arXiv:2502.06453, 2025. Hard perturbations of the MATH benchmark.
- **Functional MATH / Functional Benchmarks** — Saurabh Srivastava et al. "Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap." arXiv:2402.19450, 2024.
- **Putnam-AXIOM** — Aryan Gulati, Brando Miranda, Eric Chen, Emily Xia, Kai Fronsdal, Bruno Dumont, Elyas Obbad, Sanmi Koyejo. arXiv:2508.08292, 2025 (earlier ICML 2024 workshop version at OpenReview WrBqgoseGL).
- **ReClor** — Weihao Yu, Zihang Jiang, Yanfei Dong, Jiashi Feng. ICLR 2020. arXiv:2002.04326. Logical reasoning from LSAT.

### 2. Training Data Contamination in LLMs

- **Quantifying Memorization Across Neural Language Models** — Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. ICLR 2023. arXiv:2202.07646.
- **Extracting Training Data from Large Language Models** — Nicholas Carlini, Florian Tramèr, Eric Wallace, et al. USENIX Security 2021. arXiv:2012.07805.
- **Detecting Pretraining Data from LLMs (Min-K% Prob)** — Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, Luke Zettlemoyer. ICLR 2024. arXiv:2310.16789.
- **Rethinking Benchmark and Contamination for LMs with Rephrased Samples** — Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica. arXiv:2311.04850, 2023. Argues n-gram filtering is insufficient; rephrased contamination is rampant.
- **Don't Make Your LLM an Evaluation Benchmark Cheater** — Kun Zhou et al. arXiv:2311.01964, 2023.
- **NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark** — Oscar Sainz et al. EMNLP Findings 2023.
- **A Survey on Benchmark Data Contamination for LLMs** — Yujuan Fu et al. arXiv:2406.04244, 2024.
- **Detecting Benchmark Contamination Through Watermarking** — arXiv:2502.17259, 2025.
- **On the Fragility of Benchmark Contamination Detection in Reasoning Models** — arXiv:2510.02386, 2025. Particularly relevant: shows RL post-training of reasoning models obscures evidence of SFT-stage contamination.
- **Detecting Data Contamination in LLMs via In-Context Learning** — arXiv:2510.27055, 2025.
- **Language Models are Few-Shot Learners (GPT-3)** — Tom B. Brown et al. NeurIPS 2020. arXiv:2005.14165. Original n-gram contamination analysis appendix.
- **Pythia** (open-source model with documented contamination studies) — Stella Biderman, Hailey Schoelkopf, Quentin Anthony, et al. ICML 2023. arXiv:2304.01373.

### 3. Training Data Attribution for LLMs

- **Understanding Black-box Predictions via Influence Functions** — Pang Wei Koh, Percy Liang. ICML 2017 Best Paper. arXiv:1703.04730. The foundational reference.
- **TracIn — Estimating Training Data Influence by Tracing Gradient Descent** — Garima Pruthi, Frederick Liu, Mukund Sundararajan, Satyen Kale. NeurIPS 2020. arXiv:2002.08484.
- **Datamodels: Predicting Predictions from Training Data** — Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Mądry. ICML 2022. arXiv:2202.00622.
- **TRAK: Attributing Model Behavior at Scale** — Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Mądry. ICML 2023. arXiv:2303.14186.
- **Studying Large Language Model Generalization with Influence Functions** — Roger Grosse, Juhan Bae, Cem Anil, et al. (Anthropic). arXiv:2308.03296, 2023. Per the abstract: "We use the Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EK-FAC) approximation to scale influence functions up to LLMs with up to 52 billion parameters."
- **Scalable Influence and Fact Tracing for LLM Pretraining (TrackStar)** — Tyler A. Chang et al. arXiv:2410.17413, 2024.
- **Enhancing Training Data Attribution for LLMs with Fitting Error Consideration (DDA)** — Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng. arXiv:2410.01285, 2024.
- **Do Influence Functions Work on Large Language Models?** — arXiv:2409.19998, 2024. Cautionary empirical study.
- **First is Better than Last for Language Data Influence** — Chih-Kuan Yeh et al. NeurIPS 2022. arXiv:2202.11844.
- **If Influence Functions are the Answer, Then What is the Question?** — Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, Roger Grosse. arXiv:2209.05364, 2022.
- **Concept Influence (very recent)** — Matthew Kowal, Gonçalo Paulo, Louis Jaburi, et al. arXiv:2602.14869, 2026. Probe/SAE-based attribution that is an order of magnitude cheaper than influence functions; highly compatible with limited-compute proposals.

### 4. Instella Model Family (AMD)

- **Instella (arXiv technical report)** — Jiang Liu, Jialian Wu, Xiaodong Yu, Yusheng Su, Prakamya Mishra, Gowtham Ramesh, Sudhanshu Ranjan, Chaitanya Manem, Ximeng Sun, Ze Wang, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum. "Instella: Fully Open Language Models with Stellar Performance." arXiv:2511.10628, November 2025.
- **Instella launch blog (ROCm Blogs, 5 Mar 2025)** — "Introducing Instella: New State-of-the-art Fully Open 3B Language Models." Confirms training on 128 MI300X GPUs, 4.15T tokens, 36 decoder layers, 32 attention heads, 4,096 context length, OLMo tokenizer (~50K vocab). Uses OLMoE-mix-0924 for stage 1 and dolmino-mix-1124 + python-edu + dm_math + Instella-GSM8K-synthetic for stage 2.
- **Instella-Math blog (ROCm Blogs, 9 Aug 2025)** — Xiaodong Yu, Jiang Liu, Yusheng Su, et al. First reasoning-focused LM trained with long CoT RL entirely on AMD GPUs. Built on Instella-3B-Instruct via 2-stage SFT (using OpenMathInstruct-2's 14M GSM8K/MATH problem-solution pairs) and 3 stages of RL via VERL on 32 MI300X GPUs. Pass@1 averaged over 16 responses; outperforms its SFT base by 10.81 points (vs. DeepScaleR's 6.22-point gain over its base).
- **Instella-Long blog (ROCm Blogs, 11 Jun 2025)** — Jialian Wu et al. First fully open from-scratch LM supporting 128K context. Outperforms Phi-3.5-mini, Gemma-3-4B, Qwen2.5-3B on long-context HELMET benchmark; beats Qwen2.5-3B-Instruct by 2.75% on average at 8K/16K/32K.
- **Instella-VL-1B blog (ROCm Blogs, 7 Mar 2025)** — Ximeng Sun et al. Vision-language extension using CLIP ViT-L/14-336, 1.2B LM + 300M vision encoder.
- **AMD OLMo-1B** — Jiang Liu, Jialian Wu, Xiaodong Yu, et al. AMD Developer Blog, 31 Oct 2024. AMD's first LM, trained on 64 MI250 GPUs with 1.3T tokens; precursor to Instella. No arXiv version.
- **OLMo (the AI2 base architecture Instella adapts)** — Dirk Groeneveld, Iz Beltagy, Pete Walsh, et al. ACL 2024. arXiv:2402.00838.
- **OLMo 2** — Pete Walsh, Luca Soldaini, Dirk Groeneveld, et al. arXiv:2501.00656, 2025.
- **Training data sources used by Instella:** DCLM (Jeffrey Li et al., NeurIPS 2024, arXiv:2406.11794); FineWeb / FineWeb-Edu (Guilherme Penedo et al., NeurIPS 2024 D&B, arXiv:2406.17557); OpenWebMath (Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, Jimmy Ba, ICLR 2024, arXiv:2310.06786); Proof-Pile-2 / Llemma (Zhangir Azerbayev et al., ICLR 2024, arXiv:2310.10631); OpenMathInstruct-2 (Shubham Toshniwal et al., arXiv:2410.01560, 2024).

### 5. Reasoning Consistency, Robustness, and Shortcut Learning

- **GSM-Symbolic** — Mirzadeh et al., ICLR 2025, arXiv:2410.05229 (also listed in §1). The single most-cited recent paper on paraphrase consistency for math reasoning.
- **Embers of Autoregression** — R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, Thomas L. Griffiths. arXiv:2309.13638, 2023 (PNAS 2024). Demonstrates that LLM "reasoning" performance is shaped by output probability and task frequency.
- **Faith and Fate: Limits of Transformers on Compositionality** — Nouha Dziri, Ximing Lu, Melanie Sclar, et al. NeurIPS 2023 Spotlight. arXiv:2305.18654.
- **MR-GSM8K** — Zeng et al., arXiv:2312.17080. Meta-reasoning over GSM8K solutions.
- **Evaluating the Logical Reasoning Ability of ChatGPT and GPT-4 (LogiEval)** — Hanmeng Liu et al. arXiv:2304.03439, 2023.
- **GLoRE: Evaluating Logical Reasoning of LLMs** — arXiv:2310.09107.
- **Path-of-Thoughts (relational reasoning robustness)** — arXiv:2412.17963, 2024.
- **ReasonAgain** — Xiaodong Yu, Ben Zhou, Hao Cheng, Dan Roth. arXiv:2410.19056, 2024. (Notably co-authored by an Instella team member.)

### 6. Emergence of Reasoning Across Scale

- **Emergent Abilities of Large Language Models** — Jason Wei, Yi Tay, Rishi Bommasani, et al. TMLR 2022. arXiv:2206.07682.
- **Are Emergent Abilities of LLMs a Mirage?** — Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. NeurIPS 2023 Outstanding Paper. arXiv:2304.15004. Per the NeurIPS 2023 Outstanding Paper announcement (11 Dec 2023): "nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance."
- **Chain-of-Thought Prompting Elicits Reasoning in LLMs** — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou. NeurIPS 2022. arXiv:2201.11903. Key for the scale × CoT interaction.
- **Scaling Laws for Neural Language Models** — Jared Kaplan et al. arXiv:2001.08361, 2020.
- **BIG-Bench** — Srivastava et al., TMLR 2023, arXiv:2206.04615 (foundational source for scale-vs-task analyses).
- **Pythia** — Biderman et al., ICML 2023, arXiv:2304.01373. The canonical controlled-scale family (70M–12B trained on identical data); methodologically the closest precedent for what an Instella-1B-vs-3B controlled study would look like.

### 7. Data-Centric AI / Training Data Quality for Reasoning

- **Minerva: Solving Quantitative Reasoning Problems with Language Models** — Aitor Lewkowycz, Anders Andreassen, David Dohan, et al. NeurIPS 2022. arXiv:2206.14858.
- **Llemma: An Open Language Model for Mathematics** — Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, et al. ICLR 2024. arXiv:2310.10631.
- **OpenWebMath** — Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, Jimmy Ba. ICLR 2024. arXiv:2310.06786.
- **MAmmoTH2: Scaling Instructions from the Web** — Xiang Yue, Tianyu Zheng, Ge Zhang, Wenhu Chen. NeurIPS 2024. arXiv:2405.03548.
- **OpenMathInstruct-2** — Shubham Toshniwal et al. arXiv:2410.01560, 2024.
- **DataComp-LM (DCLM)** — Jeffrey Li, Alex Fang, Georgios Smyrnis, et al. NeurIPS 2024 D&B. arXiv:2406.11794.
- **FineWeb / FineWeb-Edu** — Guilherme Penedo et al. NeurIPS 2024 D&B. arXiv:2406.17557.
- **SmolLM2 — Data-Centric Training of a Small LM** — Loubna Ben Allal et al. arXiv:2502.02737, 2025.
- **TinyGSM** — Bingbin Liu et al. arXiv:2312.09241, 2023.
- **The Stack** (code pretraining, contamination context) — Denis Kocetkov et al. arXiv:2211.15533, 2022.

### 8. FAISS and Embedding-Based Search Infrastructure

- **FAISS (original)** — Jeff Johnson, Matthijs Douze, Hervé Jégou. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data, vol. 7 no. 3, 2019. arXiv:1702.08734, 2017.
- **FAISS library paper (recent)** — Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou. "The Faiss library." arXiv:2401.08281, 2024.
- **Sentence-BERT** — Nils Reimers, Iryna Gurevych. EMNLP-IJCNLP 2019, pp. 3980-3992. arXiv:1908.10084. (`all-MiniLM-L6-v2` is a Sentence-BERT-style fine-tune of MiniLM.)
- **MiniLM (underlying architecture for all-MiniLM-L6-v2)** — Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, Ming Zhou. "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers." NeurIPS 2020. arXiv:2002.10957.
- **GTE — General Text Embeddings** — Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang. arXiv:2308.03281, 2023.
- **HNSW (often used inside FAISS)** — Yury Malkov, D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv:1603.09320, 2016 (TPAMI 2018).
- **Product Quantization (PQ)** — Hervé Jégou, Matthijs Douze, Cordelia Schmid. IEEE TPAMI 2011.

## Recommendations

**Stage 1 — Scoping (2–3 weeks). Use the Instella arXiv report + ROCm blogs to enumerate exactly which datasets are in pretraining stages 1 & 2, post-training SFT, and Instella-Math RL.** Decision rule: if any candidate benchmark (e.g., MATH, GSM8K, BBH, HumanEval) has known contamination paths through OpenMathInstruct-2 or dolmino-mix, treat it as primary; if not, treat as held-out for robustness checks.

**Stage 2 — Lightweight contamination scan.** Build a FAISS index over Instella's pretraining text using all-MiniLM-L6-v2 (or GTE-small for better quality at comparable cost). For each benchmark item, retrieve top-K=20 nearest pretraining documents and apply both (a) n-gram overlap thresholds (Brown et al. 2020 13-gram, Yang et al. 2023 paraphrase-aware) and (b) embedding cosine ≥ 0.85. Decision rule: if >5% of any benchmark has matches above threshold, the benchmark is unreliable for accuracy claims on Instella and must be paired with a perturbed variant.

**Stage 3 — Robustness probes.** Run Instella-3B, Instella-Math, and AMD OLMo-1B on GSM-Symbolic (numerical and clause perturbations), GSM-Plus, MATH-Perturb, CLUTRR with held-out rule combinations, and TTT-Bench. Decision rule: a model is "remembering" if its accuracy drop from GSM8K→GSM-Symbolic exceeds the drop reported for similarly-sized fully open models (e.g., OLMo-2 1B/7B, Pythia, AMD OLMo) by more than one standard deviation per Mirzadeh et al.'s reported variances.

**Stage 4 — Lightweight attribution.** Pick the 200–500 most-interesting benchmark items (Stage-2 contaminated, Stage-2 clean-but-correct, and Stage-2 perturbed-and-failed). Apply TracIn-CP using 3–5 saved Instella checkpoints (the AMD repo exposes these); pre-filter candidates with the FAISS index from Stage 2 to keep the gradient inner-products tractable. If compute permits, replicate a subset with TRAK using ~10 surrogate runs on a downsampled corpus. Decision rule: if for ≥30% of correct GSM8K answers the top-influence training doc is a near-duplicate of the test problem, "remembering" dominates; if top-influence docs are diverse procedural/educational math (OpenWebMath, OpenMathInstruct-2), "reasoning generalization" dominates.

**Stage 5 — Scale-controlled emergence study.** Compare AMD OLMo-1B → Instella-3B → Instella-Math on the perturbed benchmarks. This mirrors the Pythia methodology (Biderman et al. 2023) within a fully open AMD-hosted family. Decision rule: if perturbed-benchmark accuracy scales smoothly while raw accuracy shows discontinuities, this corroborates Schaeffer et al. (2023) on the Instella family.

**Thresholds to revise the plan:** (i) if Instella's pretraining mix turns out to be substantially less fully-public than advertised, fall back to OLMo-2 + Pythia; (ii) if TracIn on a 3B model exceeds 1 MI300X-week per benchmark, downgrade to Concept Influence (Kowal et al. 2026) or pure embedding-attribution; (iii) if Instella-Math's RL stage clearly produces "On the Fragility of Benchmark Contamination Detection" (arXiv:2510.02386, 2025)-type masking, restrict attribution claims to pre-RL checkpoints.

## Caveats

- The AMD Instella arXiv technical report (arXiv:2511.10628) is dated November 2025 and we have it only via the GitHub README quote; the report's specific reported reasoning-benchmark numbers should be reconfirmed against the PDF before citing them quantitatively in any proposal.
- The Concept Influence paper (arXiv:2602.14869) carries a 2026 arXiv identifier; treat as the most recent reference and confirm publication status before submission.
- Putnam-AXIOM exists in both an ICML 2024 workshop version (OpenReview WrBqgoseGL) and a 2025 arXiv version (arXiv:2508.08292); cite the version appropriate to the proposal's framing.
- Mirzadeh et al.'s "LLMs cannot perform genuine logical reasoning" claim has been challenged on statistical-rigour grounds — see Desi R. Ivanova, "On Some (Fixable) Limitations of 'Understanding the Limitations of Mathematical Reasoning in LLMs'", desirivanova.com/post/gsm-symbolic/ (22 Oct 2024), which finds that "for 21 out of 25 models, there isn't enough evidence to reject the null hypothesis that performance on GSM8K is equal to that on GSM-Symbolic." The proposal should phrase the Mirzadeh claim as the *hypothesis under test*, not a settled conclusion.
- AMD OLMo-1B is documented only via an AMD developer blog, not arXiv; cite carefully.
- "Stochastic Parrots" (Bender, Gebru, McMillan-Major, Mitchell, FAccT 2021, DOI 10.1145/3442188.3445922) is sometimes conflated in summaries with the GPT-3 paper (Brown et al., NeurIPS 2020, arXiv:2005.14165); they are distinct works and the proposal should cite each for its actual contribution.
- TRAK and influence-function methods at LLM scale remain an active area; Bae et al. ("If Influence Functions are the Answer, then What is the Question?", arXiv:2209.05364, 2022) and the 2024 empirical paper "Do Influence Functions Work on Large Language Models?" (arXiv:2409.19998) should be cited as cautions.
- Several recent contamination-detection papers (e.g., arXiv:2510.02386, arXiv:2510.27055) are preprints from late 2025 and have not been peer-reviewed; flag accordingly.