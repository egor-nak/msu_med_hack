"""Глобальные веса фьюжна и смещения под min-F1 по вне-выборочным дампам (fusion.json).

  python tools/fit_fusion.py --run results/experiments/E08_transfer/<cfg> --out solution/artifacts \
      [--mode loglinear|stacking|avg] [--loso]

Вход — дампы прогона (dumps/<session>.csv: сглаженные лог-вероятности ЭЭГ `le`, NIRS `ln`) и истинные метки
(runs/<session>/predictions.csv). Дампы должны быть получены в LOSO-режиме (глобальная ЭЭГ-модель без
оцениваемого субъекта) — тогда вероятности вне-выборочные.

  * loglinear: w_eeg ∈ {0.6, 0.7, 0.8, 0.9, 1.0} — максимум среднего по сессиям (60·F + 30·P + 15·F_nirs);
  * stacking: multinomial LR на [le, ln] (6 признаков), C ∈ {0.1, 1} (выбор — внутренний leave-subject-out);
  * смещения b = (0, b2, b3) для гибрида, ЭЭГ и NIRS — координатный спуск по сетке [−1.5, 1.5] шаг 0.1,
    максимизация среднего по сессиям min-F1.
--loso: для каждого субъекта S всё подбирается по остальным субъектам → <out>/loso/S/fusion.json
(используется только для честной оценки); без --loso — по всем → <out>/fusion.json (финальный артефакт).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "tools"))
from hbci.fusion import calibrate_bias, fuse, min_f1_many  # noqa: E402
from common import subject_of  # noqa: E402

W_GRID = [0.6, 0.7, 0.8, 0.9, 1.0]


def load_dumps(run: Path) -> pd.DataFrame:
    parts = []
    for d in sorted((run / "dumps").glob("*.csv")):
        D = pd.read_csv(d).drop_duplicates("end", keep="last")
        p = pd.read_csv(run / "runs" / d.stem / "predictions.csv")[["end_sample", "true_y"]]
        m = p.merge(D, left_on="end_sample", right_on="end", how="inner")
        m["session"] = d.stem
        m["subject"] = subject_of(d.stem)
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def _arr(df: pd.DataFrame, pref: str) -> np.ndarray:
    return df[[f"{pref}{i}" for i in (1, 2, 3)]].to_numpy()


def _recall_many(pred: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    M = pred.shape[0]
    _, gi = np.unique(g, return_inverse=True)
    G = gi.max() + 1
    code = ((np.arange(M)[:, None] * G + gi[None, :]) * 3 + pred) * 3 + y[None, :]
    cm = np.bincount(code.ravel(), minlength=M * G * 9).reshape(M, G, 3, 3)
    col = cm.sum(2)
    rec = np.where(col > 0, np.diagonal(cm, axis1=2, axis2=3) / np.maximum(col, 1), 0.0)
    return rec.mean(2).mean(1)


def fit_stack(le: np.ndarray, ln: np.ndarray, y: np.ndarray, C: float) -> dict:
    from sklearn.linear_model import LogisticRegression
    Z = np.hstack([le, ln])
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=500, random_state=0).fit((Z - mu) / sd, y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "W": m.coef_.tolist(), "b": m.intercept_.tolist(), "C": C}


def fit_all(df: pd.DataFrame, mode: str) -> dict:
    """Подбор весов и смещений по данным df (все сессии df — обучающие)."""
    y = df.true_y.to_numpy()
    g = df.session.to_numpy()
    le = _arr(df, "le")
    has_n = ~np.isnan(_arr(df, "ln")).any(1)
    out: dict = {"mode": mode, "n_sessions": int(df.session.nunique()),
                 "train_subjects": sorted(df.subject.unique().tolist())}
    out_bias = {"eeg": calibrate_bias(le, y, g).tolist()}
    dn = df[has_n]
    if len(dn):
        yn, gn = dn.true_y.to_numpy(), dn.session.to_numpy()
        len_, lnn = _arr(dn, "le"), _arr(dn, "ln")
        out_bias["nirs"] = calibrate_bias(lnn, yn, gn).tolist()
        if mode == "loglinear":
            best = None
            for w in W_GRID:
                lh = fuse(len_, lnn, "loglinear", w)
                b = calibrate_bias(lh, yn, gn)
                pred = np.argmax(lh + b, 1)[None, :]
                sc = 60 * min_f1_many(pred, yn - 1, gn)[0] + 30 * _recall_many(pred, yn - 1, gn)[0]
                if best is None or sc > best[0] + 1e-9:
                    best = (sc, w, b)
            out["w_eeg"] = best[1]
            out_bias["hyb"] = best[2].tolist()
        elif mode == "stacking":
            subs = dn.subject.to_numpy()
            best = None
            for C in (0.1, 1.0):
                preds = np.zeros((len(dn), 3))
                for s in np.unique(subs):  # внутренний leave-subject-out для выбора C
                    tr = subs != s
                    st = fit_stack(len_[tr], lnn[tr], yn[tr], C)
                    preds[~tr] = fuse(len_[~tr], lnn[~tr], "stacking", stack=st)
                b = calibrate_bias(preds, yn, gn)
                sc = min_f1_many(np.argmax(preds + b, 1)[None, :], yn - 1, gn)[0]
                if best is None or sc > best[0]:
                    best = (sc, C)
            st = fit_stack(len_, lnn, yn, best[1])
            out["stack"] = st
            out_bias["hyb"] = calibrate_bias(fuse(len_, lnn, "stacking", stack=st), yn, gn).tolist()
        else:
            out_bias["hyb"] = calibrate_bias(fuse(len_, lnn, mode), yn, gn).tolist()
    out["bias"] = out_bias
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="loglinear", choices=["loglinear", "stacking", "avg"])
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--global-from", default=None,
                    help="каталог с global_eeg.joblib (и loso/<S>/global_eeg.joblib) — копируется рядом с fusion.json")
    a = ap.parse_args()
    import shutil
    df = load_dumps(Path(a.run))
    out = Path(a.out)
    if a.loso:
        for s in sorted(df.subject.unique()):
            r = fit_all(df[df.subject != s], a.mode)
            r["source_run"] = str(a.run)
            d = out / "loso" / s
            d.mkdir(parents=True, exist_ok=True)
            if a.global_from:
                shutil.copy(Path(a.global_from) / "loso" / s / "global_eeg.joblib", d / "global_eeg.joblib")
            (d / "fusion.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
            print(s, {k: r[k] for k in ("w_eeg",) if k in r}, r["bias"])
    else:
        r = fit_all(df, a.mode)
        r["source_run"] = str(a.run)
        out.mkdir(parents=True, exist_ok=True)
        if a.global_from and (Path(a.global_from) / "global_eeg.joblib").exists():
            shutil.copy(Path(a.global_from) / "global_eeg.joblib", out / "global_eeg.joblib")
        (out / "fusion.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(json.dumps(r, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
