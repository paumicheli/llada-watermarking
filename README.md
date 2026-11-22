# Evaluating Watermark Robustness in Diffusion Language Models

**Trustworthy AI Models and Systems — University of Bern, Autumn 2026**  
**Authors:** Paul Micheli and Kenny Gayakpa  
**Supervisor:** Roberto Gheda

This repository contains the notebooks, experimental configurations, analysis code, and selected results for our seminar project on **watermark removal from diffusion language model outputs**. We reproduce established removal attacks against the watermarking method from *Watermarking Diffusion Language Models* and investigate a new **stealing-guided WordNet substitution attack**.

Our main research question is: **Can an attacker learn statistical patterns from watermarked and unwatermarked responses to select better words for replacement, without knowing the secret watermark key?**

## Project overview

We study **LLaDA-8B-Instruct** using the diffusion-language-model watermarking implementation by Gloaguen et al. Our experiments have three stages:

1. **Data generation and detection:** generate matching watermarked (WM) and unwatermarked (NoWM) completions and reproduce local watermark detection.
2. **Attack reproduction:** evaluate word deletion, WordNet synonym substitution, BERT context-aware substitution, Qwen-based paraphrasing, and back-translation.
3. **Our attack:** learn lexical statistics from labeled WM/NoWM text, use them to select substitution positions, and compare against randomized baselines at **matched actual word-edit budgets**.

## Main findings

| Experiment | Result |
|---|---:|
| Watermarked responses correctly detected before attack | 583 / 600 (97.17%) |
| Unwatermarked responses incorrectly detected | 1 / 600 (0.17%) |
| Reproduced WordNet attack, requested ratio 0.50 | 55.9% ASR |
| Qwen paraphrasing, raw evaluation | 92.3% ASR |
| Qwen back-translation, raw evaluation | 63.6% ASR |

**ASR (attack success rate)** is the proportion of initially detected watermarked completions that become undetected after modification. Detection uses the local retokenized-text detector with threshold `p < 0.01`. The rewriting methods and word-substitution methods have different modification scopes and should not be interpreted as direct equal-budget comparisons.

For our **stealing-guided benchmark** (minimum support 1, requested word-edit ratio 0.20, ten development splits), the methods were compared at the same *realized* number of word substitutions for each response:

| Attack | Mean ASR |
|---|---:|
| Random WordNet | 10.58% |
| V1 — word unigrams | 18.21% |
| V2 — random selection within the bigram candidate pool | 19.96% |
| V2 — ranked word bigrams | 23.77% |
| V3 — ranked retokenized token bigrams | 22.54% |
| **V2 — guided synonym selection** | **25.11%** |

The mean actual word-edit ratio in this comparison was **17.84%**. The ablations indicate that a substantial part of the improvement comes from the **eligible-position pool**; smaller gains from statistical ranking and guided synonym choice should be interpreted cautiously. A Holm-corrected paired McNemar comparison on the reference development split found a difference between ranked V2 and global random WordNet (`p = 0.00245`), but not clear statistical evidence for superiority of the more elaborate variants over the corresponding pool-controlled baselines.

**These are exploratory development results, not confirmation on a completely independent benchmark.**

## Repository organization

The principal components are:

```text
configs/                                   Watermarked / unwatermarked generation settings
src/detection.py                           Local watermark detection helper
notebooks/
  02_attack_reproduction/
    04_word_deletion.ipynb                  Word deletion
    05_synonym_substitution.ipynb           WordNet substitution
    06_context_aware_substitution.ipynb     BERT contextual substitution
    07_paraphrase_qwen.ipynb                Qwen paraphrasing
    08_backtranslation_qwen.ipynb           Qwen back-translation
    09_attack_comparison.ipynb              Reproduction summary and figures
  Guided_attacks/
    experiments/
      stealing_guided_v2.ipynb              Word-bigram method and ablations
      benchmark_watermark_v2.ipynb          Final matched-budget development benchmark
    archive/                                Earlier V1/V3 and pilot investigations
results/
  attacks/
    comparison/                             Reproduced-attack summary CSVs and figures
    fair_comparison_benchmark_v2/
      development_full/                     Final benchmark summary outputs
      development_pilot/                    Earlier benchmark and diagnostics
```

> Folder names and notebook paths above describe the project organization used during the experiments. If your checkout differs, locate the notebook by its basename. Earlier notebooks may also depend on existing local input files under `results/Cleaned Data/` and `results/attacks/`.

## Attack methodology

The attacker learns relative feature frequencies from **labeled watermarked and unwatermarked text**, with no access to the watermark secret key. We examine three representations:

- **V1 — word unigrams:** score individual words.
- **V2 — word bigrams:** score adjacent word pairs and rank candidate substitution positions based on their context.
- **V3 — token bigrams:** use adjacent LLaDA-tokenizer IDs from **post-hoc tokenization of saved text**, not internal diffusion states.

Candidate words must admit WordNet replacements. The evaluation separates the benefit of selecting an **eligible pool**, **ranking positions within that pool**, and **choosing replacement synonyms**. It also matches the *actual number of replacements per response* across methods, rather than assuming a requested substitution ratio results in the same edit count.

For each development split, **400 paired examples** are used to learn the statistics and **200 paired examples** are used for evaluation. Ten overlapping splits characterize stability; they are **not ten independent external test datasets**.

## Running the experiments

These are **Jupyter notebooks**, not a single-command application. The main dependencies include Python, PyTorch, Transformers, NumPy, pandas, SciPy, NLTK/WordNet, and the packages used by the original watermarking repository. Some attack notebooks additionally require **MarkLLM**, sentence-transformer embeddings, and GPU-enabled Qwen inference.

1. Clone this repository and open it in **VS Code / Jupyter** or **Google Colab**.
2. For **generation**, follow the setup notebook and the authors' [diffusion-lm-watermark repository](https://github.com/eth-sri/diffusion-lm-watermark). Generation with LLaDA-8B-Instruct requires substantial GPU memory; our generation experiments used an NVIDIA A100.
3. For **detection and classical attacks**, prepare the paired saved completions and run the detection and reproduction notebooks in sequence.
4. For **Qwen rewriting**, use a GPU runtime and the corresponding paraphrasing/back-translation notebooks. Our reported rewriting experiments used Qwen2.5-7B-Instruct on an NVIDIA L4.
5. For **our attack**, start with `notebooks/Guided_attacks/experiments/benchmark_watermark_v2.ipynb`. Check its configuration, input paths, saved detector and tokenizer, and output-directory settings before running. The benchmark requires the paired cleaned completions and detection baseline; regenerating the underlying LLaDA texts is not necessary if these are already available.

**Reproduction caveat:** Some notebooks reference a local checkout of the original watermarking implementation and local result paths. The repository is an academic research record; it does not provide a verified, fully automated installation-and-rerun procedure from a fresh machine.

## Data, outputs, and storage

The dataset uses **600 WaterBench prompts**: 200 each from Longform QA, Finance QA, and AlpacaFarm, with WM and NoWM completions for each prompt (**1,200 completions total**).

Smaller aggregate outputs (CSVs, figures, configuration and manifest files) are included where practical. Large raw benchmark files, notably `observations.csv`, and human-evaluation answer keys are **intentionally excluded from Git tracking**. The absence of these raw files limits exact reconstruction of some per-response evaluations from the public checkout alone.

When working in Google Colab, **copy JSON/JSONL/CSV results to persistent storage (Google Drive or your local machine)** before the runtime terminates. Temporary files under `/content/` will otherwise be lost.

## Limitations

- A successful watermark-removal attack does **not automatically preserve meaning or fluency**. The Qwen experiments use an embedding-based similarity check; the matched-budget lexical benchmark does not establish equivalent semantic quality across methods.
- The ten development splits overlap, and methods were refined on this corpus. Reported statistical comparisons are exploratory.
- A post-hoc tokenization analysis (V3) should not be confused with access to LLaDA's hidden generation-time token states or watermark key.
- Results are specific to the model, watermark configuration, prompts, detector, and attack procedures used here.

## References and acknowledgments

This project builds on:

- Gloaguen et al., **[Watermarking Diffusion Language Models](https://github.com/eth-sri/diffusion-lm-watermark)** (ICLR 2026), including the official code and watermark detector.
- Nie et al., *Large Language Diffusion Models* (LLaDA).
- Tu et al., *WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models*.
- Pan et al., *MarkLLM: An Open-Source Toolkit for LLM Watermarking*.
- Miller, *WordNet: A Lexical Database for English*.
- Qwen Team, *Qwen2.5 Technical Report*.
- Reimers and Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.

**Course:** Seminar Trustworthy AI Models and Systems, University of Bern (2026).  
**Supervisor:** Roberto Gheda.
