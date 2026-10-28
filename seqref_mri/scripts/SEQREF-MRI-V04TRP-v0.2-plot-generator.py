#!/usr/bin/env python3
# SEQREF-MRI-V04TRP v0.2 -- six-run training plot generator
# LIFETIME: KEEP
"""Regenerate the eight V04TRR evidence plots from hash-verified records.

Required inputs are supplied explicitly by CLI; no path is guessed. The script
fails closed on missing files, hash drift, schema drift, or non-finite plotted
values. It writes the eight PNG files plus plot_manifest.json. No fallback,
placeholder, mock, dummy return, or silent pass is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ABBR = "SEQREF-MRI-V04TRP"
VERSION = "0.2"
SCHEMA = "seqref-mri-v04tr-plot-manifest/1"
RECORD_SCHEMA = "seqref-v04-train-record/1"
LEDGER_SCHEMA = "seqref-v04-run-ledger/1"
LAMBDA_PRIME = 2.4348938608414534
EXPECTED_COMMIT_PIN = "10cedfded01fb4080d869363661c7cf87725f0f1"
FIGURE1_ROLLING_WINDOW = 25
EXPECTED_LEDGER_SHA256 = (
    "6962c0db0b4a2e3ce54de6fe8a8669e45c9e6987c80216ba0da1234480828f89"
)
EXPECTED_RECORD_SHA256 = {
    "v04_arm0_seed101_attempt1_train_record.json":
        "f591a5efff4bf334e5ad6883c380da51d888276f700fb3c080d378d6b8d41481",
    "v04_arm1_seed101_attempt1_train_record.json":
        "d129bde2728d22235744ba8e1dbd266ae2821807345ce1c2d2275c8e43b6ad6d",
    "v04_arm0_seed202_attempt1_train_record.json":
        "5e08b27d2a7523045eee265b32159aa7f108a053418fff8bbd2c79e8e8b086c7",
    "v04_arm1_seed202_attempt1_train_record.json":
        "b917eb9ee8bf00e0ce17de5e87da2b90f06dc87de6280daf6cefa4c37fdb9689",
    "v04_arm0_seed303_attempt1_train_record.json":
        "f69cb2f6d2ced5c8ab98238c1fa0b4c0746a9851eb8742f456ae7e72fe20e125",
    "v04_arm1_seed303_attempt1_train_record.json":
        "474750df0e5dd401e429397d3a34bc3becda74f711c159218ebb9b907df324b4",
}
EXPECTED_ORDER = [(101, 0), (101, 1), (202, 0), (202, 1),
                  (303, 0), (303, 1)]
SEEDS = (101, 202, 303)
ARMS = (0, 1)
ARM_COLORS = {0: "#1f77b4", 1: "#d62728"}
SEED_STYLES = {101: "-", 202: "--", 303: ":"}
EPOCH_BOUNDARIES = (1086, 2172)

logger = logging.getLogger(ABBR)


class V04TRPError(RuntimeError):
    """Typed plot-generation failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", ABBR, code, message)
    raise V04TRPError(code, message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("INPUT_JSON_INVALID", f"cannot load {path}: {exc}")
    if not isinstance(value, dict):
        _fail("INPUT_SCHEMA_INVALID", f"{path} does not contain a JSON object")
    return value


def finite_array(values: list[Any], name: str) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        _fail("PLOT_DATA_INVALID", f"{name} is not numeric: {exc}")
    if out.size == 0 or not np.isfinite(out).all():
        _fail("PLOT_DATA_INVALID", f"{name} is empty or non-finite")
    return out


def load_verified_inputs(record_dir: Path, ledger_path: Path
                         ) -> tuple[dict[tuple[int, int], dict[str, Any]],
                                    dict[str, Any], dict[str, str]]:
    if not record_dir.is_dir():
        _fail("RECORD_DIR_MISSING", f"record directory missing: {record_dir}")
    if not ledger_path.is_file():
        _fail("LEDGER_MISSING", f"ledger missing: {ledger_path}")
    ledger_sha = sha256_file(ledger_path)
    if ledger_sha != EXPECTED_LEDGER_SHA256:
        _fail("LEDGER_HASH_MISMATCH",
              f"{ledger_path.name} sha256 {ledger_sha} != registered hash")
    ledger = load_json(ledger_path)
    if ledger.get("schema") != LEDGER_SCHEMA:
        _fail("LEDGER_SCHEMA_MISMATCH",
              f"ledger schema {ledger.get('schema')!r}")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or len(entries) != 6:
        _fail("LEDGER_ENTRIES_INVALID", "ledger must contain six entries")

    records: dict[tuple[int, int], dict[str, Any]] = {}
    input_hashes: dict[str, str] = {"v04_run_ledger.json": ledger_sha}
    for filename, expected_sha in EXPECTED_RECORD_SHA256.items():
        path = record_dir / filename
        if not path.is_file():
            _fail("RECORD_MISSING", f"required record missing: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            _fail("RECORD_HASH_MISMATCH",
                  f"{filename} sha256 {actual_sha} != registered hash")
        rec = load_json(path)
        if rec.get("schema") != RECORD_SCHEMA:
            _fail("RECORD_SCHEMA_MISMATCH",
                  f"{filename} schema {rec.get('schema')!r}")
        key = (int(rec.get("seed", -1)), int(rec.get("arm", -1)))
        if key in records:
            _fail("RECORD_DUPLICATE", f"duplicate arm/seed record {key}")
        records[key] = rec
        input_hashes[filename] = actual_sha

    if set(records) != set(EXPECTED_ORDER):
        _fail("RUN_SET_MISMATCH", f"records cover {sorted(records)}")
    ledger_order = [(int(e.get("seed", -1)), int(e.get("arm", -1)))
                    for e in entries]
    if ledger_order != EXPECTED_ORDER:
        _fail("RUN_ORDER_MISMATCH", f"ledger order is {ledger_order}")

    for seed, arm in EXPECTED_ORDER:
        rec = records[(seed, arm)]
        entry = entries[EXPECTED_ORDER.index((seed, arm))]
        expected_name = (
            f"v04_arm{arm}_seed{seed}_attempt1_train_record.json"
        )
        if entry.get("record_file") != expected_name:
            _fail("LEDGER_RECORD_NAME_MISMATCH",
                  f"{(seed, arm)} ledger file {entry.get('record_file')!r}")
        if entry.get("record_sha256") != input_hashes[expected_name]:
            _fail("LEDGER_RECORD_HASH_MISMATCH",
                  f"{(seed, arm)} ledger hash does not match the record")
        if rec.get("status") != "completed" or rec.get("attempt") != 1 \
                or rec.get("steps_completed") != 3258 \
                or rec.get("error_code") is not None:
            _fail("RUN_COMPLETION_MISMATCH",
                  f"{(seed, arm)} is not completed attempt 1 at 3258 steps")
        expected_lambda = 0.0 if arm == 0 else LAMBDA_PRIME
        if float(rec.get("lambda", math.nan)) != expected_lambda:
            _fail("LAMBDA_MISMATCH",
                  f"{(seed, arm)} lambda {rec.get('lambda')!r}")
        cfg = rec.get("config") or {}
        expected_cfg = {"optimizer": "Adam", "lr": 0.0001, "batch": 32,
                        "epochs": 3, "steps": 3258, "precision": "float32",
                        "device": "cpu", "schedule": None,
                        "gradient_clipping": None, "init_seed": seed}
        for field, expected in expected_cfg.items():
            if cfg.get(field) != expected:
                _fail("CONFIG_MISMATCH",
                      f"{(seed, arm)} config.{field}={cfg.get(field)!r}")
        if rec.get("commit_pin") != EXPECTED_COMMIT_PIN:
            _fail("COMMIT_PIN_MISMATCH",
                  f"{(seed, arm)} commit pin {rec.get('commit_pin')!r}")
        history = rec.get("history")
        vals = rec.get("validation_history")
        if not isinstance(history, list) or len(history) != 3258:
            _fail("HISTORY_LENGTH_MISMATCH",
                  f"{(seed, arm)} history is not 3258 steps")
        step_sequence = [h.get("step") for h in history]
        if step_sequence != list(range(1, 3259)):
            _fail("STEP_SEQUENCE_MISMATCH",
                  f"{(seed, arm)} history steps are not exactly 1...3258")
        if not isinstance(vals, list) or len(vals) != 3:
            _fail("VALIDATION_LENGTH_MISMATCH",
                  f"{(seed, arm)} validation history is not 3 epochs")
        epoch_sequence = [v.get("epoch") for v in vals]
        if epoch_sequence != [1, 2, 3]:
            _fail("EPOCH_SEQUENCE_MISMATCH",
                  f"{(seed, arm)} monitor epochs are {epoch_sequence!r}")
        finite_array([h["L_density"] for h in history],
                     f"{(seed, arm)} history L_density")
        finite_array([h["L_total"] for h in history],
                     f"{(seed, arm)} history L_total")
        finite_array([h["grad_norm"] for h in history],
                     f"{(seed, arm)} history grad_norm")
        finite_array([v["L_val"] for v in vals],
                     f"{(seed, arm)} validation L_val")
        finite_array([v["val_NLL"] for v in vals],
                     f"{(seed, arm)} validation val_NLL")
        finite_array([v["val_L_rec"] for v in vals],
                     f"{(seed, arm)} validation val_L_rec")
        if not math.isfinite(float(rec.get("wall_clock_s", math.nan))) \
                or float(rec["wall_clock_s"]) <= 0.0:
            _fail("RUNTIME_INVALID", f"{(seed, arm)} wall clock is invalid")
        checkpoints = rec.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 3:
            _fail("CHECKPOINT_COUNT_MISMATCH",
                  f"{(seed, arm)} checkpoint count is not 3")
        checkpoint_epochs = [c.get("epoch") for c in checkpoints]
        if checkpoint_epochs != [1, 2, 3]:
            _fail("CHECKPOINT_EPOCH_MISMATCH",
                  f"{(seed, arm)} checkpoint epochs are {checkpoint_epochs!r}")
        recorded_ckpts = [c.get("file_sha256") for c in checkpoints]
        if recorded_ckpts != entry.get("checkpoints"):
            _fail("CHECKPOINT_LEDGER_MISMATCH",
                  f"{(seed, arm)} checkpoint list differs from ledger")
    return records, ledger, input_hashes


def history_xy(rec: dict[str, Any], field: str) -> tuple[np.ndarray, np.ndarray]:
    history = rec["history"]
    return (finite_array([h["step"] for h in history], f"history step"),
            finite_array([h[field] for h in history], f"history {field}"))


def validation_xy(rec: dict[str, Any], field: str) -> tuple[np.ndarray, np.ndarray]:
    vals = rec["validation_history"]
    return (finite_array([v["epoch"] for v in vals], f"validation epoch"),
            finite_array([v[field] for v in vals], f"validation {field}"))


def style_axes(ax: plt.Axes, y_log: bool = False) -> None:
    ax.grid(True, alpha=0.25)
    if y_log:
        ax.set_yscale("log")


def save_figure(fig: plt.Figure, path: Path) -> None:
    try:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    except OSError as exc:
        _fail("PLOT_WRITE_FAILED", f"cannot write {path}: {exc}")
    finally:
        plt.close(fig)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or values.size < window:
        _fail("ROLLING_WINDOW_INVALID",
              f"window={window}, values={values.size}")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_training_loss(records: dict[tuple[int, int], dict[str, Any]],
                       out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    window = FIGURE1_ROLLING_WINDOW
    for seed in SEEDS:
        for arm in ARMS:
            rec = records[(seed, arm)]
            x, density = history_xy(rec, "L_density")
            _, total = history_xy(rec, "L_total")
            style = SEED_STYLES[seed]
            smooth_x = x[window - 1:]
            axes[0].plot(x, density, color=ARM_COLORS[arm],
                         linestyle=style, linewidth=0.35, alpha=0.18)
            axes[0].plot(smooth_x, rolling_mean(density, window),
                         color=ARM_COLORS[arm], linestyle=style,
                         linewidth=1.35,
                         label=f"arm {arm}, seed {seed} (rolling {window})")
            axes[arm + 1].plot(x, total, color=ARM_COLORS[arm],
                               linestyle=style, linewidth=0.35, alpha=0.18)
            axes[arm + 1].plot(smooth_x, rolling_mean(total, window),
                               color=ARM_COLORS[arm], linestyle=style,
                               linewidth=1.35,
                               label=f"seed {seed} (rolling {window})")
    axes[0].set_title("Density loss (common scale)")
    axes[0].set_ylabel("L_density")
    axes[1].set_title("Total objective — arm 0")
    axes[2].set_title("Total objective — arm 1")
    for ax in axes:
        ax.set_xlabel("Training step")
        style_axes(ax)
    axes[1].set_ylabel("L_total")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"V04 training loss curves (faint raw traces; rolling mean, window={window} steps)"
    )
    fig.tight_layout()
    path = out / "figure_01_training_loss_curves.png"
    save_figure(fig, path)
    return path


def plot_gradient_norms(records: dict[tuple[int, int], dict[str, Any]],
                        out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for arm in ARMS:
            x, grad = history_xy(records[(seed, arm)], "grad_norm")
            ax.plot(x, grad, color=ARM_COLORS[arm], linewidth=0.65,
                    alpha=0.85, label=f"arm {arm}")
        for boundary in EPOCH_BOUNDARIES:
            ax.axvline(boundary, color="black", linewidth=0.8, alpha=0.35)
        ax.set_title(f"Seed {seed}")
        ax.set_xlabel("Training step")
        style_axes(ax, y_log=True)
    axes[0].set_ylabel("Global gradient norm (log scale)")
    axes[0].legend()
    fig.suptitle("V04 gradient norms by seed")
    fig.tight_layout()
    path = out / "figure_02_gradient_norms.png"
    save_figure(fig, path)
    return path


def plot_monitor_lval(records: dict[tuple[int, int], dict[str, Any]],
                      out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for arm, ax in zip(ARMS, axes):
        for seed in SEEDS:
            epochs, values = validation_xy(records[(seed, arm)], "L_val")
            ax.plot(epochs, values, marker="o", linestyle=SEED_STYLES[seed],
                    color=ARM_COLORS[arm], label=f"seed {seed}")
            min_i = int(np.argmin(values))
            ax.scatter([epochs[min_i]], [values[min_i]], s=140,
                       facecolors="none", edgecolors="black", linewidths=1.4)
        ax.set_title("Arm 0 (λ=0)" if arm == 0 else "Arm 1 (λ=λ₂′)")
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 2, 3])
        style_axes(ax)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Monitor L_val")
    fig.suptitle("Monitor L_val by epoch (circles mark descriptive minima)")
    fig.tight_layout()
    path = out / "figure_03_monitor_lval.png"
    save_figure(fig, path)
    return path


def plot_monitor_metric(records: dict[tuple[int, int], dict[str, Any]],
                        out: Path, field: str, title: str,
                        ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for seed in SEEDS:
        for arm in ARMS:
            epochs, values = validation_xy(records[(seed, arm)], field)
            ax.plot(epochs, values, marker="o", linestyle=SEED_STYLES[seed],
                    color=ARM_COLORS[arm], label=f"arm {arm}, seed {seed}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 2, 3])
    style_axes(ax)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = out / filename
    save_figure(fig, path)
    return path


def final_epoch(records: dict[tuple[int, int], dict[str, Any]], seed: int,
                arm: int, field: str) -> float:
    vals = records[(seed, arm)]["validation_history"]
    if int(vals[-1]["epoch"]) != 3:
        _fail("FINAL_EPOCH_MISMATCH",
              f"{(seed, arm)} final validation epoch is {vals[-1]['epoch']!r}")
    return float(vals[-1][field])


def plot_paired_differences(records: dict[tuple[int, int], dict[str, Any]],
                            out: Path) -> Path:
    nll_pct = []
    rec_pct = []
    for seed in SEEDS:
        nll0 = final_epoch(records, seed, 0, "val_NLL")
        nll1 = final_epoch(records, seed, 1, "val_NLL")
        rec0 = final_epoch(records, seed, 0, "val_L_rec")
        rec1 = final_epoch(records, seed, 1, "val_L_rec")
        nll_pct.append(100.0 * (nll1 - nll0) / nll0)
        rec_pct.append(100.0 * (rec1 - rec0) / rec0)
    x = np.arange(len(SEEDS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.2))
    b0 = ax.bar(x - width / 2, nll_pct, width, label="NLL change",
                color=ARM_COLORS[0])
    b1 = ax.bar(x + width / 2, rec_pct, width,
                label="Reconstruction-loss change", color=ARM_COLORS[1])
    ax.axhline(0.0, color="black", linewidth=1.0)
    for bars in (b0, b1):
        for bar in bars:
            value = float(bar.get_height())
            va = "bottom" if value >= 0.0 else "top"
            offset = 0.006 if value >= 0.0 else -0.006
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                    f"{value:+.3f}%", ha="center", va=va, fontsize=8)
    ax.set_xticks(x, [f"seed {seed}" for seed in SEEDS])
    ax.set_ylabel("Arm 1 vs arm 0 at epoch 3 (%)")
    ax.set_title("Paired final-epoch differences")
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    path = out / "figure_06_paired_final_epoch_differences.png"
    save_figure(fig, path)
    return path


def diagnostic_values(records: dict[tuple[int, int], dict[str, Any]],
                      field: str) -> tuple[list[str], list[float], list[int]]:
    labels: list[str] = []
    values: list[float] = []
    arms: list[int] = []
    for seed in SEEDS:
        for arm in ARMS:
            diag = records[(seed, arm)].get(
                "post_training_conditioning_diagnostic")
            if not isinstance(diag, dict) or diag.get("gating") is not False:
                _fail("DIAGNOSTIC_INVALID",
                      f"{(seed, arm)} diagnostic missing or gating drifted")
            value = float(diag[field])
            if not math.isfinite(value):
                _fail("DIAGNOSTIC_INVALID",
                      f"{(seed, arm)} diagnostic {field} is non-finite")
            labels.append(f"s{seed}\na{arm}")
            values.append(value)
            arms.append(arm)
    return labels, values, arms


def plot_conditioning(records: dict[tuple[int, int], dict[str, Any]],
                      out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, field, title, ylabel in (
            (axes[0], "abs_change", "Absolute output change",
             "Absolute change"),
            (axes[1], "rel_change", "Relative output change",
             "Relative change")):
        labels, values, arms = diagnostic_values(records, field)
        ax.bar(np.arange(len(values)), values,
               color=[ARM_COLORS[a] for a in arms])
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        style_axes(ax)
    fig.suptitle(
        "Non-gating conditioning diagnostic (same z, different conditioning)"
    )
    fig.tight_layout()
    path = out / "figure_07_conditioning_diagnostic.png"
    save_figure(fig, path)
    return path


def plot_runtime(records: dict[tuple[int, int], dict[str, Any]],
                 out: Path) -> Path:
    runtimes = {
        arm: np.asarray([
            float(records[(seed, arm)]["wall_clock_s"]) / 60.0
            for seed in SEEDS
        ], dtype=np.float64)
        for arm in ARMS
    }
    if not np.isfinite(runtimes[0]).all() \
            or not np.isfinite(runtimes[1]).all():
        _fail("RUNTIME_INVALID", "runtime series contains non-finite values")
    overhead = (runtimes[1] / runtimes[0] - 1.0) * 100.0
    mean_overhead = (float(runtimes[1].mean())
                     / float(runtimes[0].mean()) - 1.0) * 100.0
    x = np.arange(len(SEEDS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12, 6.8))
    bars0 = ax.bar(x - width / 2, runtimes[0], width,
                   label="Arm 0 (λ=0)", color=ARM_COLORS[0])
    bars1 = ax.bar(x + width / 2, runtimes[1], width,
                   label="Arm 1 (λ=λ₂′)", color=ARM_COLORS[1])
    for bars in (bars0, bars1):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=9)
    for xi, value in zip(x, overhead):
        ax.text(xi + width / 2, max(runtimes[1]) + 3.8,
                f"+{value:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x, [f"seed {seed}" for seed in SEEDS])
    ax.set_ylabel("Wall-clock time (minutes)")
    ax.set_title("Observed runtime by paired seed")
    ax.set_ylim(0.0, float(max(runtimes[1])) * 1.20)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.99, 0.96, f"Mean overhead: {mean_overhead:.2f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "#666666",
                  "alpha": 0.9, "boxstyle": "round,pad=0.35"})
    fig.text(0.01, 0.01,
             "Recorded wall-clock only; host load and diagnostic overhead may contribute.",
             ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = out / "figure_08_runtime_by_seed.png"
    save_figure(fig, path)
    return path


def generate_plots(records: dict[tuple[int, int], dict[str, Any]],
                   out_dir: Path, overwrite: bool) -> list[Path]:
    try:
        if out_dir.exists():
            if not out_dir.is_dir():
                _fail("OUTPUT_DIR_INVALID", f"not a directory: {out_dir}")
            if any(out_dir.iterdir()) and not overwrite:
                _fail("OUTPUT_DIR_NOT_EMPTY",
                      f"{out_dir} is not empty; use --overwrite explicitly")
        else:
            out_dir.mkdir(parents=True)
    except OSError as exc:
        _fail("OUTPUT_DIR_FAILED", f"cannot create {out_dir}: {exc}")
    generated = [
        plot_training_loss(records, out_dir),
        plot_gradient_norms(records, out_dir),
        plot_monitor_lval(records, out_dir),
        plot_monitor_metric(records, out_dir, "val_NLL",
                            "Monitor NLL by epoch", "Monitor NLL",
                            "figure_04_monitor_nll.png"),
        plot_monitor_metric(records, out_dir, "val_L_rec",
                            "Monitor reconstruction loss by epoch",
                            "Monitor reconstruction loss",
                            "figure_05_monitor_reconstruction.png"),
        plot_paired_differences(records, out_dir),
        plot_conditioning(records, out_dir),
        plot_runtime(records, out_dir),
    ]
    expected_names = [
        "figure_01_training_loss_curves.png",
        "figure_02_gradient_norms.png",
        "figure_03_monitor_lval.png",
        "figure_04_monitor_nll.png",
        "figure_05_monitor_reconstruction.png",
        "figure_06_paired_final_epoch_differences.png",
        "figure_07_conditioning_diagnostic.png",
        "figure_08_runtime_by_seed.png",
    ]
    if [p.name for p in generated] != expected_names:
        _fail("PLOT_SET_MISMATCH", "generated plot set/order drifted")
    for path in generated:
        if not path.is_file() or path.stat().st_size == 0:
            _fail("PLOT_WRITE_FAILED", f"empty or missing output {path}")
    return generated


def write_manifest(out_dir: Path, script_path: Path,
                   records: dict[tuple[int, int], dict[str, Any]],
                   input_hashes: dict[str, str], generated: list[Path],
                   overwrite: bool) -> Path:
    environments = sorted({
        json.dumps(records[key]["environment"], sort_keys=True)
        for key in EXPECTED_ORDER
    })
    try:
        font_path = font_manager.findfont("DejaVu Sans",
                                          fallback_to_default=False)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    except Exception as exc:
        _fail("FONT_RESOLUTION_FAILED", f"cannot resolve DejaVu Sans: {exc}")
    manifest = {
        "schema": SCHEMA,
        "lifetime": "KEEP",
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "script": {
            "abbr": ABBR,
            "version": VERSION,
            "file": script_path.name,
            "sha256": sha256_file(script_path),
        },
        "inputs": input_hashes,
        "plot_settings": {
            "backend": "Agg",
            "dpi": 140,
            "arm_colors": {str(k): v for k, v in ARM_COLORS.items()},
            "seed_linestyles": {str(k): v for k, v in SEED_STYLES.items()},
            "epoch_boundaries": list(EPOCH_BOUNDARIES),
            "figure_1_smoothing": {
                "method": "rolling_mean",
                "window_steps": FIGURE1_ROLLING_WINDOW,
                "raw_traces": "retained at low alpha",
            },
            "figure_2_y_axis": "log",
            "figure_6_delta": "100 * (arm1 - arm0) / arm0 at final epoch",
        },
        "rendering_environment": {
            "python_version": platform.python_version(),
            "matplotlib_version": matplotlib.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "font": {
                "requested_family": "DejaVu Sans",
                "resolved_family": font_name,
                "resolved_path": str(font_path),
            },
        },
        "recorded_environments": [json.loads(x) for x in environments],
        "output_overwrite": {
            "overwrite_allowed": bool(overwrite),
            "policy": "fresh output directory required unless --overwrite is explicit",
        },
        "outputs": [
            {"file": path.name, "sha256": sha256_file(path),
             "bytes": path.stat().st_size}
            for path in generated
        ],
    }
    path = out_dir / "plot_manifest.json"
    try:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    except OSError as exc:
        _fail("MANIFEST_WRITE_FAILED", f"cannot write {path}: {exc}")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate V04TRR evidence plots from verified records")
    parser.add_argument("--record-dir", required=True,
                        help="directory containing the six train records")
    parser.add_argument("--ledger", required=True,
                        help="path to v04_run_ledger.json")
    parser.add_argument("--out-dir", required=True,
                        help="fresh directory for eight PNGs and plot_manifest.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into a non-empty output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        records, _, input_hashes = load_verified_inputs(
            Path(args.record_dir), Path(args.ledger))
        generated = generate_plots(records, Path(args.out_dir),
                                   bool(args.overwrite))
        manifest = write_manifest(Path(args.out_dir), Path(__file__).resolve(),
                                  records, input_hashes, generated,
                                  bool(args.overwrite))
    except V04TRPError as exc:
        logger.error("[%s] ERROR: %s", ABBR, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected plot-generation error", ABBR)
        return 2
    print(f"PASS {ABBR} v{VERSION}: 8 plots + {manifest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Changelog
# v0.2 (2026-10-15): pins the implementation commit and validates exact step,
# monitor-epoch, checkpoint-count, and checkpoint-epoch sequences fail-closed.
# Improves Figure 1 with faint raw traces plus 25-step rolling means.
# Requires a fresh output directory unless --overwrite is explicit.
# v0.1 (2026-10-15): initial hash-verified eight-figure generator and manifest.
