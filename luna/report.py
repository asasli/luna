"""Report generation for LUNA anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


def generate_anomaly_report(
    scores: dict[str, np.ndarray],
    ensemble_score: np.ndarray,
    flags: np.ndarray,
    meta: list[dict] | None = None,
    p_value: np.ndarray | None = None,
    class_names: list[str] | None = None,
    logits: np.ndarray | None = None,
    family_flags: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Generate per-sample anomaly report as a DataFrame.

    Args:
        scores: {method_name: (N,) scores}
        ensemble_score: (N,) combined anomaly score
        flags: (N,) boolean flags
        p_value: (N,) conformal p-values (optional)
        meta: list of per-sample metadata dicts (optional)
        class_names: class name strings (optional)
        logits: (N, C) logits for computing predictions/confidence (optional)
        family_flags: {family: (N,) bool} per-family detection flags (optional)

    Returns:
        DataFrame with one row per sample, sorted by ensemble_score descending.
    """
    N = len(ensemble_score)
    report = {}

    # Metadata columns
    if meta is not None:
        report["obj_id"] = [m.get("obj_id", f"sample_{i}") for i, m in enumerate(meta)]
        report["true_class"] = [m.get("class", "?") for m in meta]

    # Classification columns (if logits available)
    if logits is not None:
        probs = softmax(logits, axis=1)
        preds = probs.argmax(axis=1)
        if class_names is not None:
            report["pred_class"] = [class_names[p] for p in preds]
        else:
            report["pred_class"] = preds
        report["confidence"] = probs.max(axis=1).round(3)

    # Ensemble score and flag
    report["ens_score"] = ensemble_score.round(4)
    report["flagged"] = flags
    if p_value is not None:
        report["p_value"] = np.asarray(p_value).round(4)

    # Per-family flags
    if family_flags is not None:
        for fam, fflags in family_flags.items():
            report[f"flag_{fam}"] = fflags
        fam_mat = np.stack(list(family_flags.values()), axis=0)
        report["n_families_caught"] = fam_mat.sum(axis=0).astype(int)

    # Selected raw scores
    for m, s in scores.items():
        report[f"score_{m}"] = s.round(3)

    df = pd.DataFrame(report)
    df = df.sort_values("ens_score", ascending=False).reset_index(drop=True)
    return df


def generate_category_ablation(
    family_analysis: dict[str, dict],
) -> pd.DataFrame:
    """Generate per-family AUROC ablation table.

    Args:
        family_analysis: output of LUNAPipeline.family_analysis()

    Returns:
        DataFrame with columns: family, AUROC, Det@FAR, Det%.
    """
    rows = []
    for fam, info in family_analysis.items():
        rows.append({
            "family": fam,
            "AUROC": round(info["auroc"], 4),
            "Det@FAR": info["detected"],
            "Det%": round(info["det_rate"] * 100, 1),
            **{f"Det% @ {100 * a:g}% FAR": round(100 * v, 1)
               for a, v in info.get("det_rate_by_far", {}).items()},
        })
    return pd.DataFrame(rows)


def write_latex_table(
    report_df: pd.DataFrame,
    out_path: str | Path,
    top_k: int = 15,
) -> None:
    """Write a compact LaTeX table of the top-K most anomalous detected samples.

    Args:
        report_df: full anomaly report DataFrame.
        out_path: output .tex file path.
        top_k: number of top samples to include.
    """
    out_path = Path(out_path)

    # Select columns that exist
    preferred_cols = [
        "obj_id", "true_class", "pred_class", "confidence",
        "n_families_caught", "ens_score", "flagged",
    ]
    # Add score columns
    score_cols = [c for c in report_df.columns if c.startswith("score_")]
    # Keep first 4 score columns to avoid table overflow
    preferred_cols.extend(score_cols[:4])

    cols = [c for c in preferred_cols if c in report_df.columns]
    top = report_df[report_df.get("flagged", True)].head(top_k)[cols].copy()

    with open(out_path, "w") as f:
        f.write(top.to_latex(
            index=False,
            float_format="%.3f",
            escape=False,
            longtable=False,
            caption=(
                f"Top-{top_k} most anomalous detected samples. "
                "``\\#fam'' is the number of OOD score families that "
                "flagged the sample independently, and "
                "``Ens.'' is the combined ensemble score."
            ),
            label="tab:anomaly_top",
        ))
