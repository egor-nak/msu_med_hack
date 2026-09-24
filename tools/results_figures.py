"""Фигуры результатов 6–12 (§7) по каталогам прогонов results/experiments/*.

SPEC задаёт ключевые пайплайны (подпись → каталог прогона) и шаги «водопада» абляций.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from make_figures import CLS_NAMES, INK, INK2, SERIES, _save, plt  # noqa: E402

EXP = ROOT / "results" / "experiments"
SPEC_FILE = ROOT / "tools" / "grids" / "figure_spec.json"
SPEC: dict = json.loads(SPEC_FILE.read_text()) if SPEC_FILE.exists() else {}


def _extra(run: str) -> pd.DataFrame:
    return pd.read_csv(EXP / run / "per_session_extra.csv")


def _boot_ci(v: np.ndarray, subj: np.ndarray, n: int = 2000) -> tuple[float, float]:
    sys.path.insert(0, str(ROOT / "tools"))
    from analyze import boot_indices
    reps = boot_indices(subj, n)
    m = [np.nanmean(v[i]) for i in reps]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fig_confusions(run: str) -> None:
    per = {p.stem: json.loads(p.read_text()) for p in sorted((EXP / run / "per_session").glob("*.json"))}
    f = {k: v["hybrid"]["min_f1"] for k, v in per.items()}
    order = sorted(f, key=f.get)
    nonirs = [k for k in order if not per[k].get("nirs_only")]
    picks = [("лучшая", order[-1]), ("медианная", order[len(order) // 2]), ("худшая", order[0])]
    if nonirs:
        picks.append(("без NIRS", nonirs[0]))
    fig, axs = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.4))
    for ax, (lab, k) in zip(axs, picks):
        G = np.array(per[k]["hybrid"]["confusion"], dtype=float)  # строки — ответы, столбцы — истина
        R = G / G.sum(0, keepdims=True)
        ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(G[i, j])}\n{R[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if R[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), CLS_NAMES)
        ax.set_yticks(range(3), CLS_NAMES)
        ax.set_xlabel("истинный класс")
        ax.set_ylabel("ответ")
        ax.grid(False)
        ax.set_title(f"{lab}: {k.split('_')[-1]}\nmin-F1 {f[k]:.2f}", fontsize=8)
    fig.suptitle("Матрицы ошибок гибрида (доля от истинного класса)")
    _save(fig, "confusion_examples.png")


def fig_blocks(runs: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (lab, run) in enumerate(runs.items()):
        d = _extra(run)
        m = [d[f"hyb_minf1_b{b}"].mean() for b in range(2, 8)]
        ax.plot(range(2, 8), m, marker="o", ms=5, color=SERIES[i], label=lab)
    ax.set_xlabel("блок (модель обучена на блоках 1…b−1)")
    ax.set_ylabel("min-F1 гибрида (среднее по сессиям)")
    ax.set_title("Кривая обучения по блокам")
    ax.legend(fontsize=7)
    _save(fig, "learning_curve_blocks.png")


def fig_time(runs: dict[str, str], chance: float | None = None) -> None:
    bins = ["1.0", "2.5", "5.0"]
    lab_bins = ["1.0–2.5 с", "2.5–5 с", "5–9 с"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    w = 0.8 / len(runs)
    for i, (lab, run) in enumerate(runs.items()):
        d = _extra(run)
        for a, (key, ttl) in enumerate([("hyb_minf1_t", "min-F1 гибрида"), ("eeg_auc_rest_im_t", "AUC покой/воображение (ЭЭГ)"),
                                        ("eeg_auc_lr_t", "AUC левая/правая (ЭЭГ)")]):
            vals = [d[f"{key}{b}"].mean() if f"{key}{b}" in d else np.nan for b in bins]
            axs[a].bar(np.arange(3) + i * w, vals, width=w * 0.92, color=SERIES[i], label=lab)
            axs[a].set_title(ttl)
    for a in range(3):
        axs[a].set_xticks(np.arange(3) + w * (len(runs) - 1) / 2, lab_bins)
        axs[a].set_xlabel("конец окна от начала сегмента")
    for a in (1, 2):
        axs[a].axhline(0.5, color=INK2, lw=1, ls="--")
        axs[a].set_ylim(0.3, 1.0)
    if chance is not None:
        axs[0].axhline(chance, color=INK2, lw=1, ls="--")
        axs[0].text(2.4, chance + 0.01, "эмпир. шанс", fontsize=7, color=INK2, ha="right")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(runs), fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Качество по времени внутри сегмента (ранние окна — до нарастания ФЭС)", y=1.02)
    _save(fig, "time_in_segment.png")


def fig_pipelines(runs: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(1.3 * len(runs) + 2, 4))
    rng = np.random.default_rng(0)
    for i, (lab, run) in enumerate(runs.items()):
        d = _extra(run)
        v = d["hyb_min_f1"].to_numpy()
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False,
                   medianprops={"color": INK2}, boxprops={"color": INK2}, whiskerprops={"color": INK2},
                   capprops={"color": INK2})
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=SERIES[i % 8], alpha=0.7, lw=0)
        lo, hi = _boot_ci(v, d["subject"].to_numpy())
        ax.errorbar(i + 0.33, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="D", color=INK, ms=4, capsize=3)
        ax.text(i + 0.33, hi + 0.015, f"{v.mean():.3f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(runs)), list(runs), rotation=20, ha="right")
    ax.set_ylabel("min-F1 гибрида по сессиям")
    ax.set_title("Сравнение пайплайнов: сессии (точки), среднее и 95 % бутстрап-ДИ (ромб)")
    _save(fig, "pipelines_comparison.png")


def fig_waterfall(steps: list[list[str]]) -> None:
    labs, vals = [], []
    for lab, run in steps:
        m = json.loads((EXP / run / "metrics.json").read_text())
        labs.append(lab)
        vals.append(m["hybrid"]["min_f1"])
    fig, ax = plt.subplots(figsize=(1.0 * len(steps) + 2.5, 4))
    prev = None
    for i, v in enumerate(vals):
        if prev is None:
            ax.bar(i, v, color=SERIES[0], width=0.6)
        else:
            d = v - prev
            ax.bar(i, d, bottom=prev, color=SERIES[1] if d < 0 else SERIES[2], width=0.6)
            ax.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color=INK2, lw=0.8)
        ax.text(i, max(v, prev or 0) + 0.004, f"{v:.3f}", ha="center", fontsize=7, color=INK)
        prev = v
    ax.set_xticks(range(len(labs)), labs, rotation=25, ha="right")
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.03)
    ax.set_ylabel("$F_{global}$ (min-F1 гибрида)")
    ax.set_title("Вклад шагов: от MVP к финалу (оранжевый — снижение, бирюзовый — рост)")
    _save(fig, "ablation_waterfall.png")


def fig_controls(final: str, heog: str, fes_db: pd.Series | None) -> None:
    from scipy.stats import spearmanr
    a = _extra(final).set_index("session")
    b = _extra(heog).set_index("session")
    n = 2 if fes_db is not None else 1
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 3.8), squeeze=False)
    ax = axs[0, 0]
    x, y = b.loc[a.index, "hyb_auc_lr"], a["eeg_auc_lr"]
    r = spearmanr(x, y, nan_policy="omit")
    ax.scatter(x, y, s=14, color=SERIES[0])
    ax.set_xlabel("AUC Л/П контрольного ГЭОГ-декодера")
    ax.set_ylabel("AUC Л/П финальной модели (ЭЭГ)")
    ax.set_title(f"Финал vs ГЭОГ-декодер: Spearman ρ = {r.statistic:.2f} (p = {r.pvalue:.2g})", fontsize=8)
    if fes_db is not None:
        ax = axs[0, 1]
        x = fes_db.reindex(a.index)
        y = a["eeg_auc_rest_im"]
        r = spearmanr(x, y, nan_policy="omit")
        ax.scatter(x, y, s=14, color=SERIES[1])
        ax.set_xlabel("сила ФЭС в сессии, дБ (81.5–85.5 Гц, вообр. / покой)")
        ax.set_ylabel("AUC покой/воображение финальной модели (ЭЭГ)")
        ax.set_title(f"Финал vs сила ФЭС: Spearman ρ = {r.statistic:.2f} (p = {r.pvalue:.2g})", fontsize=8)
    _save(fig, "controls_scatter.png")


def fig_modalities(run: str) -> None:
    d = _extra(run)
    d = d[d["has_nirs"]]
    cols = [("ЭЭГ", "eeg_min_f1"), ("NIRS", "nirs_min_f1"), ("гибрид", "hyb_min_f1")]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    rng = np.random.default_rng(0)
    for i, (lab, c) in enumerate(cols):
        v = d[c].to_numpy()
        axs[0].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=SERIES[i], alpha=0.6, lw=0)
        lo, hi = _boot_ci(v, d["subject"].to_numpy())
        axs[0].errorbar(i + 0.3, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="D", color=INK, ms=4, capsize=3)
        axs[0].text(i + 0.3, hi + 0.015, f"{v.mean():.3f}", ha="center", fontsize=7)
    axs[0].set_xticks(range(3), [c[0] for c in cols])
    axs[0].set_ylabel("min-F1 по сессиям")
    axs[0].set_title(f"Модальности, одинаковый порядок обучения ({len(d)} сессий с NIRS)")
    for i, (lab, c) in enumerate(cols):
        f1 = [d[f"{c.split('_')[0]}_f1_{k}"].mean() for k in (1, 2, 3)]
        axs[1].bar(np.arange(3) + i * 0.27, f1, width=0.25, color=SERIES[i], label=lab)
    axs[1].set_xticks(np.arange(3) + 0.27, CLS_NAMES)
    axs[1].set_ylabel("F1 класса (среднее по сессиям)")
    axs[1].set_title("F1 по классам")
    axs[1].legend(fontsize=7)
    _save(fig, "modalities.png")


def draw_all(spec: dict) -> None:
    if not spec:
        print("нет tools/grids/figure_spec.json — фигуры результатов пропущены")
        return
    fig_confusions(spec["final"])
    fig_blocks(spec["blocks"])
    fig_time(spec["time"], spec.get("chance"))
    fig_pipelines(spec["pipelines"])
    fig_waterfall(spec["waterfall"])
    fes = None
    p = ROOT / "results" / "cache" / "fes_strength.csv"
    if p.exists():
        fes = pd.read_csv(p).set_index("session")["fes_db"]
    fig_controls(spec["final"], spec["heog"], fes)
    fig_modalities(spec["final"])
