"""Сводная таблица всех экспериментов → results/summary.md.

  python tools/collect_summary.py

Для каждого results/experiments/<exp>/<cfg>/: F, P, NIRS F, EEG F, балл, 95 % бутстрап-ДИ балла,
режим оценки (LOSO / сессионный / in-sample), тайминги. Балл здесь — без штрафов (эксперименты идут
параллельно, -j 8, и тайминги завышены конкуренцией за CPU); честные тайминги — только прогоны -j 1
(results/metrics и E12_final_loso), они приведены отдельно.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, score_from  # noqa: E402

EXP = ROOT / "results" / "experiments"


def row(d: Path) -> dict | None:
    m = d / "metrics.json"
    if not m.exists():
        return None
    M = json.loads(m.read_text())
    C = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    X = json.loads((d / "metrics_extra.json").read_text()) if (d / "metrics_extra.json").exists() else {}
    cfg = C.get("cfg", {})
    h, n = M["hybrid"], M.get("nirs_only") or {}
    nf = n.get("min_f1")
    uses_global = bool(cfg.get("global", {}).get("enabled")) or cfg.get("bias", {}).get("mode") in ("global",) \
        or (cfg.get("fusion", {}).get("mode") == "stacking")
    if C.get("loso"):
        mode = "LOSO"
    elif uses_global:
        mode = "in-sample (глоб.)"
    else:
        mode = "сессионный"
    s = pd.read_csv(d / "summary.csv") if (d / "summary.csv").exists() else None
    ci = X.get("ci95", {}).get("score", [np.nan, np.nan])
    ms = X.get("mean_over_sessions", {})
    return {
        "эксперимент": d.parent.name, "конфигурация": d.name,
        "F": h["min_f1"], "P": h["macro_recall"], "NIRS F": nf if nf is not None else np.nan,
        "EEG F": M.get("eeg_only_info", {}).get("min_f1", np.nan),
        "балл": score_from(h["min_f1"], h["macro_recall"], nf),
        "ДИ балла": f"[{ci[0]:.1f}; {ci[1]:.1f}]" if np.isfinite(ci[0]) else "",
        "AUC покой/вообр. (ЭЭГ)": ms.get("eeg_auc_rest_im", np.nan), "AUC Л/П (ЭЭГ)": ms.get("eeg_auc_lr", np.nan),
        "оценка": mode,
        "fit max, с": float(s.fit_max_s.max()) if s is not None else np.nan,
        "predict max, с": float(s.predict_max_s.max()) if s is not None else np.nan,
    }


def main() -> None:
    rows = [r for d in sorted(EXP.glob("*/*")) if d.is_dir() for r in [row(d)] if r]
    df = pd.DataFrame(rows)
    lines = ["# Сводка экспериментов", "",
             "Онлайн-симуляция `run.py`+`score.py` на 56 сессиях train (невзвешенное среднее по сессиям).",
             "Балл = 60·F + 30·P + (5 + 15·NIRS F) **без штрафов**: эксперименты шли параллельно (`-j 8`), тайминги",
             "в них завышены конкуренцией за CPU и приведены справочно. Честные тайминги (`-j 1`) — в",
             "`results/metrics/` и `results/experiments/E12_final_loso/`. ДИ — 95 % бутстрап (субъекты → сессии, 2000).",
             "", "Режим оценки: **LOSO** — глобальные компоненты обучены без оцениваемого субъекта; **сессионный** — только",
             "онлайн-дообучение по прошедшим блокам (глобальных компонентов нет, оценка честная); **in-sample (глоб.)** —",
             "глобальные компоненты видели оцениваемого субъекта (завышено).", ""]
    for exp, g in df.groupby("эксперимент", sort=True):
        lines += [f"## {exp}", "", g.drop(columns=["эксперимент"]).to_markdown(index=False, floatfmt=".4f"), ""]
    out = ROOT / "results" / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(ROOT / "results" / "summary_all.csv", index=False, float_format="%.4f")
    print("→", out, len(df), "строк")


if __name__ == "__main__":
    main()
