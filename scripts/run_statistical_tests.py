#!/usr/bin/env python3
"""
Statistical grounding for the CrowS-Pairs bias evaluation.

Analyses
--------
1. Two-proportion z-test
   H0: stereotype_score(base) == stereotype_score(lora)
   Uses the raw per-pair preference indicators extracted from both models.

2. Pearson correlation (per model)
   Correlates the embedding-norm difference
       norm_diff = ||e_more|| - ||e_less||
   with cosine *dissimilarity* between the two sentences
       cos_dissim = 1 - cosine_similarity(e_more, e_less)
   A significant positive correlation would mean the model assigns higher
   embedding salience to the sentence that is MORE different from its
   paired counterpart — informative about representational geometry.

Outputs
-------
results/statistical_tests/
    report.txt          — human-readable summary
    results.json        — machine-readable numbers
    correlation_plot.png — scatter + regression line for both models

Usage (GPU cluster)
-------------------
    python scripts/run_statistical_tests.py --device cuda --batch-size 64

Usage (CPU / debug)
-------------------
    python scripts/run_statistical_tests.py --device cpu --batch-size 8
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# ── project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.extract_embeddings import EmbeddingExtractor
from src.evaluation.crows_pairs_eval import load_crows_pairs, _compute_pair_scores
from src.utils.logger import get_logger

logger = get_logger("statistical_tests")

CROWS_CSV = PROJECT_ROOT / "data" / "CrowS" / "crows_pairs_anonymized.csv"
OUT_DIR   = PROJECT_ROOT / "results" / "statistical_tests"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Two-proportion z-test on stereotype scores
# ─────────────────────────────────────────────────────────────────────────────
def two_proportion_ztest(
    base_prefers_stereo: np.ndarray,
    lora_prefers_stereo: np.ndarray,
) -> dict:
    """
    Two-tailed two-proportion z-test.

    Null hypothesis : p_base == p_lora
    (where p = proportion of pairs where the model prefers the stereo sentence)

    Parameters
    ----------
    base_prefers_stereo : bool array (N,)
    lora_prefers_stereo : bool array (N,)

    Returns
    -------
    dict with z_stat, p_value, base_ss, lora_ss, n_pairs
    """
    n = len(base_prefers_stereo)
    count_base = int(base_prefers_stereo.sum())
    count_lora = int(lora_prefers_stereo.sum())

    # proportions_ztest expects counts and nobs as arrays when comparing two props
    z_stat, p_value = proportions_ztest(
        count=[count_base, count_lora],
        nobs=[n, n],
        alternative="two-sided",
    )

    return {
        "test": "two-proportion z-test (base SS vs LoRA SS)",
        "n_pairs": n,
        "base_count": count_base,
        "lora_count": count_lora,
        "base_ss": round(count_base / n, 6),
        "lora_ss": round(count_lora / n, 6),
        "z_statistic": round(float(z_stat), 6),
        "p_value": round(float(p_value), 8),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Pearson correlation: norm-diff vs cosine dissimilarity
# ─────────────────────────────────────────────────────────────────────────────
def pearson_norm_vs_dissimilarity(
    scores: dict,
    model_label: str,
) -> dict:
    """
    Pearson r between:
        X = norm(e_more) - norm(e_less)   [norm difference]
        Y = 1 - cosine_sim(e_more, e_less) [cosine dissimilarity]

    Parameters
    ----------
    scores : dict returned by _compute_pair_scores
    model_label : str  e.g. "base" or "lora"

    Returns
    -------
    dict with r, p_value, n
    """
    norm_diff   = scores["norm_more"] - scores["norm_less"]          # X
    cos_dissim  = 1.0 - scores["cosine_sims"]                        # Y

    r, p_value = stats.pearsonr(norm_diff, cos_dissim)

    return {
        "test": f"Pearson correlation — norm_diff vs cosine_dissimilarity ({model_label})",
        "model": model_label,
        "n_pairs": len(norm_diff),
        "pearson_r": round(float(r), 6),
        "p_value": round(float(p_value), 8),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlations(
    base_scores: dict,
    lora_scores: dict,
    save_path: Path,
) -> None:
    """Side-by-side scatter plots with regression lines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, scores, label, colour in zip(
        axes,
        [base_scores, lora_scores],
        ["Base MiniLM", "LoRA MiniLM"],
        ["#4C72B0", "#DD8452"],
    ):
        x = scores["norm_more"] - scores["norm_less"]
        y = 1.0 - scores["cosine_sims"]

        ax.scatter(x, y, s=8, alpha=0.35, color=colour, edgecolors="none")

        # Regression line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
                label=f"slope={m:.4f}")

        r, p = stats.pearsonr(x, y)
        ax.set_title(f"{label}\nr = {r:.4f},  p = {p:.2e}", fontsize=12)
        ax.set_xlabel("Norm Difference  (||e_more|| − ||e_less||)")
        ax.set_ylabel("Cosine Dissimilarity  (1 − sim)")
        ax.legend(fontsize=9)

    plt.suptitle(
        "Embedding Norm Difference vs Cosine Dissimilarity\n(CrowS-Pairs, N=1508)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation plot → %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_report(ztest: dict, pearson_base: dict, pearson_lora: dict) -> str:
    sig = lambda d: "YES" if d["significant_at_0.05"] else "NO"
    lines = [
        "=" * 65,
        "  STATISTICAL TESTS — CrowS-Pairs Bias Evaluation",
        "=" * 65,
        "",
        "─── Test 1: Two-Proportion Z-Test (Base SS vs LoRA SS) ─────────",
        f"  N pairs          : {ztest['n_pairs']}",
        f"  Base SS           : {ztest['base_ss']:.4f}  ({ztest['base_count']} / {ztest['n_pairs']})",
        f"  LoRA SS           : {ztest['lora_ss']:.4f}  ({ztest['lora_count']} / {ztest['n_pairs']})",
        f"  z-statistic       : {ztest['z_statistic']:.4f}",
        f"  p-value           : {ztest['p_value']:.6f}",
        f"  Significant α=0.05: {sig(ztest)}",
        f"  Significant α=0.01: {'YES' if ztest['significant_at_0.01'] else 'NO'}",
        "",
        "  Interpretation: The difference in stereotype scores between the",
        "  base model and LoRA model is statistically significant, providing",
        "  strong evidence that fine-tuning on StereoSet shifts the model's",
        "  embedding-space preferences." if ztest["significant_at_0.05"] else
        "  Interpretation: The difference is NOT statistically significant.",
        "",
        "─── Test 2a: Pearson Correlation — Base MiniLM ─────────────────",
        f"  Pearson r         : {pearson_base['pearson_r']:.4f}",
        f"  p-value           : {pearson_base['p_value']:.6f}",
        f"  Significant α=0.05: {sig(pearson_base)}",
        "",
        "─── Test 2b: Pearson Correlation — LoRA MiniLM ─────────────────",
        f"  Pearson r         : {pearson_lora['pearson_r']:.4f}",
        f"  p-value           : {pearson_lora['p_value']:.6f}",
        f"  Significant α=0.05: {sig(pearson_lora)}",
        "",
        "  Interpretation: A positive r indicates that pairs with greater",
        "  norm disparity between stereo/anti-stereo sentences also tend to",
        "  be less similar (more semantically distant) in the embedding space.",
        "=" * 65,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading CrowS-Pairs …")
    df = load_crows_pairs(CROWS_CSV)
    sent_more = df["sent_more"].tolist()
    sent_less = df["sent_less"].tolist()

    # ── Extract embeddings ────────────────────────────────────────────────────
    logger.info("Loading base model …")
    base_ext = EmbeddingExtractor("base", device=args.device)
    logger.info("Extracting base embeddings …")
    base_emb_more = base_ext.extract(sent_more, batch_size=args.batch_size)
    base_emb_less = base_ext.extract(sent_less, batch_size=args.batch_size)
    base_scores   = _compute_pair_scores(base_emb_more, base_emb_less)
    del base_ext  # free GPU memory before loading LoRA

    logger.info("Loading LoRA model …")
    lora_ext = EmbeddingExtractor("lora", device=args.device)
    logger.info("Extracting LoRA embeddings …")
    lora_emb_more = lora_ext.extract(sent_more, batch_size=args.batch_size)
    lora_emb_less = lora_ext.extract(sent_less, batch_size=args.batch_size)
    lora_scores   = _compute_pair_scores(lora_emb_more, lora_emb_less)
    del lora_ext

    # ── Run tests ────────────────────────────────────────────────────────────
    logger.info("Running two-proportion z-test …")
    ztest_result = two_proportion_ztest(
        base_scores["prefers_stereo"],
        lora_scores["prefers_stereo"],
    )

    logger.info("Running Pearson correlation (base) …")
    pearson_base = pearson_norm_vs_dissimilarity(base_scores, "base")

    logger.info("Running Pearson correlation (lora) …")
    pearson_lora = pearson_norm_vs_dissimilarity(lora_scores, "lora")

    # ── Save outputs ─────────────────────────────────────────────────────────
    report = _fmt_report(ztest_result, pearson_base, pearson_lora)

    report_path = OUT_DIR / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved report → %s", report_path)

    json_path = OUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "two_proportion_ztest": ztest_result,
                "pearson_base": pearson_base,
                "pearson_lora": pearson_lora,
            },
            f,
            indent=2,
        )
    logger.info("Saved JSON → %s", json_path)

    plot_correlations(base_scores, lora_scores, OUT_DIR / "correlation_plot.png")

    # ── Print to stdout ───────────────────────────────────────────────────────
    print("\n" + report)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical tests for CrowS-Pairs bias evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (cuda / cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
