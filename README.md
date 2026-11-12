# Trustworthy AI — Watermark Removal Attacks on Diffusion Language Models

University seminar project (Trustworthy AI). We reproduce part of the experiments of
*Watermarking Diffusion Language Models* (Gloaguen, Staab, Jovanović, Vechev — ICLR 2026)
and study removal attacks against this watermark, with an emphasis on preserving meaning.

- Paper: https://arxiv.org/abs/2509.24368
- Official code: https://github.com/eth-sri/diffusion-lm-watermark
- Related attack paper: Chainwash, https://arxiv.org/abs/2605.05503

## Project status

| Stage | Status |
|---|---|
| Generation (600 watermarked + 600 unwatermarked, LLaDA-8B-Instruct) | Done |
| Data cleaning and prompt alignment | Done |
| Baseline detection (local, retokenized text) | Done |
| Extended baseline (green fraction, scored tokens, length, per-dataset TPR) | TODO |
| Reproduction of the original robustness attacks (paper Fig. 3) | TODO |
| Semantic-preservation evaluation of attacks | TODO |
| New removal attacks | TODO |

## Dependency on the official implementation

We do not modify the official watermark or detector. The official repository is used as an
external dependency, pinned to the commit we inspected:

```
eth-sri/diffusion-lm-watermark @ 380a4bef75879c28ebcfaf954f1974deb0d54980
```

TODO: confirm which commit was actually installed during the historical generation runs.
If it cannot be established, state that here instead of assuming it.

## Experimental configuration

Generation: `GSAI-ML/LLaDA-8B-Instruct`, bfloat16, CUDA (Colab A100), steps 300,
gen_length 300, block_length 25, temperature 0.5, remasking random, cfg_scale 0.0, chat true,
batch_size 1.

Watermark (`watermark_type: Ours`): delta 4.0, convolution_kernel [-1], topk 50,
seeding_scheme sumhash, greenlist_type bernoulli, gamma 0.25, enforce_kl false, n_iter 1.
The green-list seed (42) is hard-coded in the official `OurWatermark` class.

Note: the official sample config `configs/main/Llada/ourWatermark_llada8b_instruct.yaml`
uses delta 2.0. Our delta 4.0 matches the paper's recommended setting for C = {-1} (Table 1).

Detection threshold: `p_value < 0.01` (one-sided binomial test, null green rate gamma = 0.25).

Random seeds for the historical generation runs were not recorded. All future experiments
record their seeds.

## Data

Three WaterBench subsets (taken from the official ETH repository, `data/WaterBench/`), the
first 200 prompts each:

| Dataset | File | Prompts |
|---|---|---:|
| Longform QA (ELI5) | 2-1_longform_qa.jsonl | 200 |
| Finance QA | 2-2_finance_qa.jsonl | 200 |
| AlpacaFarm | 5-1_alpacafarm.jsonl | 200 |

```
results/
    No watermark Data/      30 raw JSONL batches (never modified)
    Watermark Data/         30 raw JSONL batches (never modified)
    Cleaned Data/
        unwatermarked_600.jsonl
        watermarked_600.jsonl
    detection_results.jsonl
```

Cleaning: 60 duplicate records were found, all in the unwatermarked AlpacaFarm batches 05, 08
and 09 (40 records for 20 prompts each). They are most likely caused by the official code
opening the output file in append mode when a batch is rerun. We keep the first completion
for each duplicated prompt. The raw files remain unchanged.

## Notebooks

| Notebook | Runs on | Purpose |
|---|---|---|
| `setup.ipynb` | Colab GPU | Environment, batch creation, watermarked / unwatermarked generation |
| `03_prepare_results.ipynb` | Mac (local) | Loading, deduplication, alignment checks, export of cleaned data |
| `04_watermark_detection.ipynb` | Mac (local, CPU) | Generation-time vs local detection, TPR / FPR |

## Baseline results (local detection on retokenized text)

| Metric | Value |
|---|---:|
| TPR (583 / 600) | 97.17% |
| FPR (1 / 600) | 0.17% (95% Wilson CI: 0.03%–0.94%) |
| Accuracy | 98.50% |

Generation-time detection (raw token IDs, the protocol behind the paper's Table 1):
576 / 600 = 96.00% (95% Wilson CI: 94.1%–97.3%). The paper reports 0.99 for this
setting; the gap is not yet explained (a likely candidate is response length).

## Methodological decisions

1. **Use one detection procedure for attack comparisons.** All attack experiments use local
   detection on retokenized text, for both original and attacked texts. This matches the
   official robustness script (`scripts/run_robustness_eval.py`), which also retokenizes
   text. Generation-time scores are used only when comparing against the paper's Table 1.
2. **Why the two detection methods differ.** Generation-time detection runs on all 300
   generated positions, including trailing special/EOS tokens. Saved completions are decoded
   with `skip_special_tokens=True`, so those tokens are absent after retokenization.
3. **Tokenizer call.** We call the tokenizer with `add_special_tokens=False`; the official
   robustness script uses the tokenizer default. TODO: check whether this adds a BOS token.
4. **What is recorded for every detection.** p-value, green fraction, number of scored tokens
   and length. Attacks that shorten text raise the p-value without removing the signal.

## Reproducibility

- Python 3.12, local `.venv`; dependency snapshot in `requirements-local.txt` (`pip freeze`).
- Never overwrite raw data or previous attack outputs; every run writes a new file.
- API keys live in `.env`, which is never committed.