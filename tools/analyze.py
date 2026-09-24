"""Расширенный анализ прогона (каталог с runs/ и per_session/; опционально dumps/).

  python tools/analyze.py results/experiments/E04_riemann/ts_slda
  python tools/analyze.py RUN_A --compare RUN_B      # парный Wilcoxon по субъектам + бутстрап-ДИ разницы

Считает (кроме официальных метрик):
  * F1 по классам (гибрид / ЭЭГ / NIRS);
  * метрики по блокам 2…7;
  * метрики по времени конца окна от начала сегмента: [1.0, 2.5), [2.5, 5), [5, 9.1] с;
  * бинарные AUC «покой vs воображение» и «Л vs П» по лог-вероятностям из dumps/;
  * агрегаты: среднее по сессиям, среднее по субъектам, 95 % бутстрап-ДИ (субъекты → сессии, 2000).
Выход: metrics_extra.json, per_session_extra.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FS, f1s, confusion, macro_recall, min_f1, score_from, session_info,  # noqa: E402
                    subject_of, window_context)

TIME_BINS = [(1.0, 2.5), (2.5, 5.0), (5.0, 9.1)]
BLOCKS = [2, 3, 4, 5, 6, 7]
N_BOOT = 2000


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_aucs(L: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """L: (N,3) лог-вероятности. AUC(воображение vs покой), AUC(П vs Л) среди окон воображения."""
    if L is None or np.isnan(L).any():
        return float("nan"), float("nan")
    s_im = np.logaddexp(L[:, 1], L[:, 2]) - L[:, 0]
    a1 = _auc(s_im[y > 1], s_im[y == 1])
    s_lr = L[:, 2] - L[:, 1]
    a2 = _auc(s_lr[y == 3], s_lr[y == 2])
    return a1, a2


def session_extra(run_dir: Path, stem: str, info: dict) -> dict:
    p = pd.read_csv(run_dir / "runs" / stem / "predictions.csv")
    y = p.true_y.to_numpy()
    blk, tin = window_context(p.end_sample.to_numpy(), info["segments"])
    out: dict = {"session": stem, "subject": subject_of(stem), "has_nirs": info["has_nirs"]}
    cols = {"hyb": "y", "eeg": "y_eeg", "nirs": "y_nirs"}
    for key, col in cols.items():
        if col not in p or p[col].isna().all():
            continue
        yp = p[col].fillna(-1).to_numpy().astype(int)
        f = f1s(confusion(y, yp))
        out[f"{key}_min_f1"] = float(f.min())
        out[f"{key}_recall"] = macro_recall(y, yp)
        for c in range(3):
            out[f"{key}_f1_{c + 1}"] = float(f[c])
        if key == "nirs":
            continue
        for b in BLOCKS:
            m = blk == b
            if m.any():
                out[f"{key}_minf1_b{b}"] = min_f1(y[m], yp[m])
                out[f"{key}_recall_b{b}"] = macro_recall(y[m], yp[m])
        for lo, hi in TIME_BINS:
            m = (tin >= lo) & (tin < hi)
            if m.any():
                out[f"{key}_minf1_t{lo}"] = min_f1(y[m], yp[m])
                out[f"{key}_recall_t{lo}"] = macro_recall(y[m], yp[m])
                out[f"{key}_acc_t{lo}"] = float(np.mean(yp[m] == y[m]))
    d = run_dir / "dumps" / f"{stem}.csv"
    if d.exists():
        D = pd.read_csv(d).drop_duplicates("end", keep="last").set_index("end")
        D = D.reindex(p.end_sample.to_numpy())
        for key, pref in (("hyb", "lh"), ("eeg", "le"), ("nirs", "ln"), ("sess", "ls"), ("glob", "lg")):
            L = D[[f"{pref}{i}" for i in (1, 2, 3)]].to_numpy()
            if np.isnan(L).all():
                continue
            ok = ~np.isnan(L).any(1)
            a1, a2 = binary_aucs(L[ok], y[ok])
            out[f"{key}_auc_rest_im"] = a1
            out[f"{key}_auc_lr"] = a2
            if key in ("hyb", "eeg"):
                for lo, hi in TIME_BINS:
                    m = ok & (tin >= lo) & (tin < hi)
                    a1, a2 = binary_aucs(L[m], y[m])
                    out[f"{key}_auc_rest_im_t{lo}"] = a1
                    out[f"{key}_auc_lr_t{lo}"] = a2
    return out


def _agg_score(df: pd.DataFrame) -> float:
    nf = df["nirs_min_f1"].mean() if "nirs_min_f1" in df and df["nirs_min_f1"].notna().any() else None
    return score_from(df["hyb_min_f1"].mean(), df["hyb_recall"].mean(), nf)


_BOOT_CACHE: dict = {}


def boot_indices(subjects: np.ndarray, n: int = N_BOOT, seed: int = 0) -> list[np.ndarray]:
    """Индексы бутстрап-выборок: ресэмплинг субъектов, затем сессий внутри субъекта."""
    key = (tuple(subjects), n, seed)
    if key not in _BOOT_CACHE:
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(subjects == s) for s in np.unique(subjects)]
        reps = []
        for _ in range(n):
            reps.append(np.concatenate([groups[gi][rng.integers(0, len(groups[gi]), len(groups[gi]))]
                                        for gi in rng.integers(0, len(groups), len(groups))]))
        _BOOT_CACHE[key] = reps
    return _BOOT_CACHE[key]


def bootstrap(df: pd.DataFrame, fn, n: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95 % ДИ статистики fn(df) (субъекты → сессии)."""
    reps = boot_indices(df["subject"].to_numpy(), n, seed)
    vals = [fn(df.iloc[i]) for i in reps]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def analyze_run(run_dir: Path, quiet: bool = False) -> dict:
    run_dir = Path(run_dir)
    info = session_info()
    stems = sorted(p.name for p in (run_dir / "runs").iterdir() if (p / "predictions.csv").exists())
    df = pd.DataFrame([session_extra(run_dir, s, info[s]) for s in stems])
    df.to_csv(run_dir / "per_session_extra.csv", index=False, float_format="%.4f")
    num = df.select_dtypes("number").columns
    by_sess = df[num].mean().to_dict()
    by_subj = df.groupby("subject")[list(num)].mean().mean().to_dict()
    res = {
        "n_sessions": len(df),
        "mean_over_sessions": {k: float(v) for k, v in by_sess.items()},
        "mean_over_subjects": {k: float(v) for k, v in by_subj.items()},
        "score": _agg_score(df),
        "ci95": {
            "hyb_min_f1": bootstrap(df, lambda d: d["hyb_min_f1"].mean()),
            "hyb_recall": bootstrap(df, lambda d: d["hyb_recall"].mean()),
            "score": bootstrap(df, _agg_score),
        },
    }
    if "nirs_min_f1" in df and df["nirs_min_f1"].notna().any():
        res["ci95"]["nirs_min_f1"] = bootstrap(df, lambda d: d["nirs_min_f1"].mean())
    (run_dir / "metrics_extra.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    if not quiet:
        m = res["mean_over_sessions"]
        print(f"{run_dir}: F={m['hyb_min_f1']:.4f} {res['ci95']['hyb_min_f1']} P={m['hyb_recall']:.4f} "
              f"score={res['score']:.2f} {res['ci95']['score']}")
    return res


def compare(run_a: Path, run_b: Path, key: str = "hyb_min_f1") -> dict:
    """Парное сравнение A − B: Wilcoxon по субъектам (средние по сессиям субъекта) + бутстрап-ДИ разницы."""
    from scipy.stats import wilcoxon
    a = pd.read_csv(Path(run_a) / "per_session_extra.csv").set_index("session")
    b = pd.read_csv(Path(run_b) / "per_session_extra.csv").set_index("session")
    common = a.index.intersection(b.index)
    d = pd.DataFrame({"subject": a.loc[common, "subject"], "diff": a.loc[common, key] - b.loc[common, key]})
    subj = d.groupby("subject")["diff"].mean()
    w = wilcoxon(subj.values) if len(subj) > 1 and np.any(subj.values != 0) else None
    ci = bootstrap(d, lambda x: x["diff"].mean())
    return {"key": key, "mean_diff_sessions": float(d["diff"].mean()), "mean_diff_subjects": float(subj.mean()),
            "n_subjects_better": int((subj > 0).sum()), "n_subjects": int(len(subj)),
            "wilcoxon_p": float(w.pvalue) if w is not None else float("nan"), "ci95_diff": ci}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compare")
    ap.add_argument("--key", default="hyb_min_f1")
    a = ap.parse_args()
    analyze_run(Path(a.run_dir))
    if a.compare:
        print(json.dumps(compare(Path(a.run_dir), Path(a.compare), a.key), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
