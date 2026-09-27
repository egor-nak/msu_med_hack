"""Диагностика признаков векторизатора коллеги (CSP log-var, лог-мощность, асимметрия) по сессиям.

  python tools/csp_diagnostics.py

Для каждой сессии и каждого признака — AUC внутри сессии «покой vs воображение» и «левая vs правая»
(окна 1 с, непересекающиеся, из results/cache/csp_windows/*). Для CSP коллеги (вход raw) и для
LOSO-CSP (вход 4–30 Гц; фильтры субъекта S обучены без S). Плюс доля мощности 50 Гц в CSP-компонентах
на сырых данных. Выход — results/experiments/E13_csp/diagnostics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "tools"))
from analyze import _auc  # noqa: E402
from hbci.csp import vectorize_windows  # noqa: E402

SYM = [("c3", "c4"), ("f3", "f4"), ("p3", "p4"), ("c7", "c8")]
CH = ["c3", "cz", "c4", "f3", "fz", "f4", "p3", "pz", "p4", "c7", "c8"]
NAMES = [f"csp{i + 1}" for i in range(8)] + [f"logP_{c}" for c in CH] + [f"asym_{a}-{b}" for a, b in SYM]
ART = ROOT / "results" / "artifacts_exp"


def aucs(F: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ri = np.array([_auc(F[y > 1, j], F[y == 1, j]) for j in range(F.shape[1])])
    lr = np.array([_auc(F[y == 3, j], F[y == 2, j]) for j in range(F.shape[1])])
    return ri, lr


def main() -> None:
    col = joblib.load(ART / "csp_colleague" / "csp_filters.joblib")
    res: dict = {}
    for tag, cache, src in (("colleague_raw", "raw_s4", None), ("loso_raw", "raw_s4", "csp_raw"),
                            ("loso_bp4-30", "bp4-30_s4", "csp_bp4-30")):
        RI, LR = [], []
        for p in sorted((ROOT / "results" / "cache" / "csp_windows" / cache).glob("*.npz")):
            z = np.load(p)
            W, y = z["W"].astype(np.float64), z["y"].astype(int)
            filt = col["filters"] if src is None else joblib.load(
                ART / src / "loso" / p.stem.split("_")[-1] / "csp_filters.joblib")["filters"]
            F = vectorize_windows(W, filt, CH, SYM)
            ri, lr = aucs(F, y)
            RI.append(ri)
            LR.append(lr)
        RI, LR = np.array(RI), np.array(LR)
        res[tag] = {n: {"auc_rest_im_median": float(np.median(RI[:, j])),
                        "abs_dev_rest_im": float(np.median(np.abs(RI[:, j] - 0.5))),
                        "auc_lr_median": float(np.median(LR[:, j])),
                        "abs_dev_lr": float(np.median(np.abs(LR[:, j] - 0.5)))} for j, n in enumerate(NAMES)}
        print(f"== {tag}: медиана |AUC−0.5| по сессиям")
        for grp, sl in (("CSP", slice(0, 8)), ("logP", slice(8, 19)), ("asym", slice(19, 23))):
            print(f"   {grp:5s} покой/вообр.: {np.round(np.median(np.abs(RI[:, sl] - 0.5), 0), 3)}")
            print(f"   {grp:5s} Л/П         : {np.round(np.median(np.abs(LR[:, sl] - 0.5), 0), 3)}")
    out = ROOT / "results" / "experiments" / "E13_csp" / "diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("→", out)


if __name__ == "__main__":
    main()
