"""Shared helpers for the removal-attack notebooks.

The detector configuration is identical to notebooks/03_watermark_detection.ipynb.
Runs locally on Mac (CPU). The 8B generation model is not needed.
"""
import json
import sys
from pathlib import Path

import torch

ALPHA = 0.01  # detection threshold: watermark detected if p_value < ALPHA

DETECTOR_PARAMS = dict(
    delta=4.0,
    convolution_kernel=[-1],
    topk=50,
    seeding_scheme="sumhash",
    greenlist_type="bernoulli",
    greenlist_params={"gamma": 0.25},
    enforce_kl=False,
    n_iter=1,
)


def load_detector(root: Path):
    """Load the LLaDA tokenizer and the official ETH detector from the local clone."""
    sys.path.insert(0, str(root / "diffusion-lm-watermark" / "src"))
    from transformers import AutoTokenizer
    from dlm_watermark.watermarks.diffusion_watermark import OurWatermark

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    detector = OurWatermark(tokenizer=tokenizer, device="cpu", **DETECTOR_PARAMS)
    return tokenizer, detector


def detect(text: str, tokenizer, detector) -> dict:
    """Detect the watermark in a saved text (retokenized, no special tokens).

    Note: for the binomial test, the official field "z_score" is the observed
    green-token fraction k/n, not a z-score. We rename it green_fraction.
    """
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    n_tokens = int(tokens.shape[1])

    with torch.no_grad():
        result = detector.detect(tokens)

    if "p_value" not in result:  # text too short for detection
        return {"p_value": None, "detected": None, "green_fraction": None,
                "n_scored": 0, "n_tokens": n_tokens}

    p_value = float(result["p_value"])
    return {
        "p_value": p_value,
        "detected": p_value < ALPHA,
        "green_fraction": float(result["z_score"]),
        "n_scored": int(sum(result["mask"])),
        "n_tokens": n_tokens,
    }


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(records: list, path: Path) -> None:
    """Save records; refuse to overwrite an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} already exists. Use a new file name.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")