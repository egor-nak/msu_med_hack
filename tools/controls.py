"""Контрольные эксперименты §5: «модель декодирует моторное представление, а не артефакты».

  python tools/controls.py perm   --cfg CFG.json [--loso DIR] [--n 50]   # §5.7 пермутационный нуль
  python tools/controls.py strat  --run RUN_DIR                           # §5.6 стратификация по ФЭС
  python tools/controls.py corr   --run RUN_DIR --heog RUN --fes RUN      # §5.3 корреляции с артефактными декодерами
  python tools/controls.py neuro  --run RUN_DIR                           # §5.9 нейрофизиологическая согласованность
  python tools/controls.py haufe  [--cfg CFG.json]                        # §5.4 паттерны Haufe (log-power + sLDA, B2 CSP)

Результаты → results/experiments/E11_controls/*.json (+ фигура паттернов).
Метки и границы сегментов здесь используются только для анализа/оценки.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from common import CACHE, DATA, FS, ROOT, macro_recall, min_f1, segments_of, subject_of  # noqa: E402

OUT = ROOT / "results" / "experiments" / "E11_controls"


# ------------------------------------------------------------- §5.7 пермутации
def permute_lr(states: np.ndarray, blocks: np.ndarray, rng) -> np.ndarray:
    """(a) Л↔П: случайная перестановка меток сегментов воображения внутри каждого блока."""
    st = states.copy()
    for g_blk in np.unique(blocks):
        segs = [g for g in segments_of(states, blocks) if g["block"] == g_blk and g["cls"] > 1]
        labs = rng.permutation([g["cls"] for g in segs])
        for g, c in zip(segs, labs):
            st[g["start"] : g["end"]] = c
    return st


def permute_phase(states: np.ndarray, blocks: np.ndarray, rng) -> np.ndarray:
    """(b) покой/воображение: в каждом блоке с вероятностью ½ метки сдвигаются на 1 сегмент
    (покой ↔ воображение местами; порядок Л/П сохраняется). Чередование сохраняется, но связь
    «сигнал ↔ покой/воображение» между блоками рвётся. (Сдвиг на 2 позиции, как в ТЗ, оставил бы все
    метки покоя на сегментах покоя и нуля для «покой/воображение» не дал бы.)"""
    st = states.copy()
    for g_blk in np.unique(blocks):
        if rng.random() < 0.5:
            continue
        segs = [g for g in segments_of(states, blocks) if g["block"] == g_blk]
        labs = [g["cls"] for g in segs]
        new = labs[1:] + labs[:1]
        for g, c in zip(segs, new):
            st[g["start"] : g["end"]] = c
    return st


def _perm_job(args) -> dict:
    path, kind, seed, cfg, loso = args
    os.environ["OC_CFG"] = cfg
    if loso:
        os.environ["HBCI_LOSO_DIR"] = loso
    import io_utils as iu
    from sim import simulate
    s = iu.load_session(path)
    if kind != "real":
        rng = np.random.default_rng(seed)
        s = dict(s)
        s["states"] = (permute_lr if kind == "lr" else permute_phase)(s["states"], s["blocks"], rng)
    out = simulate(s)
    ends = np.array([e for e, _ in out])
    y = s["states"][ends - 1]
    res = {"session": Path(path).stem, "kind": kind, "seed": seed}
    for key in ("y", "y_eeg", "y_nirs"):
        p = np.array([r.get(key) if r.get(key) is not None else -1 for _, r in out])
        if (p > 0).any():
            res[f"{key}_minf1"] = min_f1(y, p)
            res[f"{key}_recall"] = macro_recall(y, p)
    return res


def cmd_perm(a) -> None:
    cfg = Path(a.cfg).read_text()
    files = sorted(DATA.glob("*.mat"))
    one_per_subj = {}
    for f in files:
        one_per_subj.setdefault(subject_of(f.stem), f)
    sess = [str(f) for f in one_per_subj.values()]
    loso = str(Path(a.loso).resolve()) if a.loso else None
    jobs = [(p, "real", 0, cfg, loso) for p in sess]
    jobs += [(p, k, i, cfg, loso) for k in ("lr", "phase") for i in range(a.n) for p in sess]
    with Pool(a.j) as pool:
        R = pool.map(_perm_job, jobs, chunksize=1)
    df = pd.DataFrame(R)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "permutations.csv", index=False, float_format="%.4f")
    real = df[df.kind == "real"]
    summ = {"sessions": [Path(p).stem for p in sess], "n_perm": a.n}
    for kind in ("lr", "phase"):
        d = df[df.kind == kind].groupby("seed").mean(numeric_only=True)
        for m in ("y_minf1", "y_recall", "y_eeg_minf1"):
            null = d[m].to_numpy()
            r = float(real[m].mean())
            summ[f"{kind}_{m}"] = {"real": r, "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
                                   "null_max": float(null.max()),
                                   "p_value": float((1 + np.sum(null >= r)) / (1 + len(null)))}
    (OUT / "permutations.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


# -------------------------------------------------------- §5.6 стратификация ФЭС
def cmd_strat(a) -> None:
    fes = pd.read_csv(CACHE / "fes_segments.csv")
    run = Path(a.run)
    rows = []
    for sess, g in fes.groupby("session"):
        p = run / "runs" / sess / "predictions.csv"
        if not p.exists():
            continue
        P = pd.read_csv(p)
        for _, seg in g[g.cls > 1].iterrows():
            m = (P.start_sample >= seg.start) & (P.end_sample <= seg.end)
            if m.sum() == 0:
                continue
            q = P[m]
            early = q[(q.end_sample - seg.start) / FS < 2.5]
            rows.append({"session": sess, "cls": seg.cls, "block": seg.block, "fes_db": seg.fes_db,
                         "recall": float((q.y == seg.cls).mean()), "recall_eeg": float((q.y_eeg == seg.cls).mean()),
                         "recall_early": float((early.y == seg.cls).mean()) if len(early) else np.nan,
                         "imag_detect": float((q.y > 1).mean())})
    df = pd.DataFrame(rows)
    df["group"] = np.where(df.fes_db < 2, "слабый <2 дБ", np.where(df.fes_db > 6, "сильный >6 дБ", "средний"))
    res = df.groupby("group")[["recall", "recall_eeg", "recall_early", "imag_detect"]].mean()
    res["n"] = df.groupby("group").size()
    # парное сравнение по сессиям (есть обе группы)
    pv = df[df.group != "средний"].groupby(["session", "group"])["recall"].mean().unstack().dropna()
    from scipy.stats import wilcoxon
    w = wilcoxon(pv.iloc[:, 0], pv.iloc[:, 1]) if len(pv) > 5 else None
    out = {"by_group": res.round(4).reset_index().to_dict("records"), "n_sessions_paired": int(len(pv)),
           "paired_mean": pv.mean().round(4).to_dict(), "wilcoxon_p": float(w.pvalue) if w else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fes_strat_{run.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ------------------------------------------------ §5.3 корреляции с контролями
def cmd_corr(a) -> None:
    from scipy.stats import linregress, spearmanr
    f = pd.read_csv(Path(a.run) / "per_session_extra.csv").set_index("session")
    out = {}
    for name, run in (("conf_heog", a.heog), ("conf_fes", a.fes)):
        c = pd.read_csv(Path(run) / "per_session_extra.csv").set_index("session").reindex(f.index)
        for mf, mc in (("hyb_min_f1", "hyb_min_f1"), ("eeg_auc_lr", "hyb_auc_lr"), ("eeg_auc_rest_im", "hyb_auc_rest_im")):
            r = spearmanr(f[mf], c[mc], nan_policy="omit")
            out[f"{name}:{mf}~{mc}"] = {"spearman_r": float(r.statistic), "p": float(r.pvalue)}
        lr = linregress(c["hyb_min_f1"], f["hyb_min_f1"])
        out[f"{name}:regression_minf1"] = {"slope": lr.slope, "intercept": lr.intercept, "r2": lr.rvalue ** 2,
                                           "p": lr.pvalue}
    fes = pd.read_csv(CACHE / "fes_strength.csv").set_index("session")["fes_db"].reindex(f.index)
    for m in ("hyb_min_f1", "eeg_auc_rest_im"):
        r = spearmanr(f[m], fes, nan_policy="omit")
        out[f"fes_strength~{m}"] = {"spearman_r": float(r.statistic), "p": float(r.pvalue)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"corr_{Path(a.run).name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ---------------------------------------------- §5.9 нейрофизиология по сессиям
def _neuro_one(path: str) -> dict:
    import io_utils as iu
    from scipy.signal import butter, sosfiltfilt
    s = iu.load_session(path)
    names = iu.EEG_CHANNEL_NAMES
    X = s["eeg"][:, [names.index("c3"), names.index("c4")]].astype(float)
    X = X - np.median(X, 0)
    A = sosfiltfilt(butter(4, [8, 13], "bandpass", fs=FS, output="sos"), X, axis=0) ** 2
    segs = segments_of(s["states"], s["blocks"])
    lat = {2: [], 3: []}
    rest, early = [], []
    for g in segs:
        w = slice(g["start"] + 250, g["end"])
        v = np.log(A[w].mean(0))
        if g["cls"] in (2, 3):
            lat[g["cls"]].append(v[0] - v[1])
            early.append(np.log(A[g["start"] + 125 : g["start"] + 625].mean(0)).mean())
        else:
            rest.append(v.mean())
    from analyze import _auc
    return {"session": Path(path).stem, "mu_lat_auc": _auc(np.array(lat[2]), np.array(lat[3])),
            "early_erd_db": float(10 / np.log(10) * (np.mean(early) - np.mean(rest)))}


def cmd_neuro(a) -> None:
    from scipy.stats import spearmanr
    with Pool(8) as pool:
        N = pd.DataFrame(pool.map(_neuro_one, [str(f) for f in sorted(DATA.glob("*.mat"))])).set_index("session")
    f = pd.read_csv(Path(a.run) / "per_session_extra.csv").set_index("session")
    out = {"mu_lat_auc_median": float(N.mu_lat_auc.median()), "early_erd_db_mean": float(N.early_erd_db.mean())}
    for x, y in (("mu_lat_auc", "eeg_auc_lr"), ("mu_lat_auc", "hyb_min_f1"), ("early_erd_db", "eeg_auc_rest_im"),
                 ("early_erd_db", "hyb_min_f1")):
        xx = N[x] if x != "mu_lat_auc" else (N[x] - 0.5).abs()
        r = spearmanr(xx.reindex(f.index), f[y], nan_policy="omit")
        out[f"{x}~{y}"] = {"spearman_r": float(r.statistic), "p": float(r.pvalue)}
    N.to_csv(CACHE / "neuro_session.csv", float_format="%.4f")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"neuro_{Path(a.run).name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["perm", "strat", "corr", "neuro", "haufe"])
    ap.add_argument("--cfg")
    ap.add_argument("--loso")
    ap.add_argument("--run")
    ap.add_argument("--heog")
    ap.add_argument("--fes")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "haufe":
        import haufe
        haufe.main()
        return
    {"perm": cmd_perm, "strat": cmd_strat, "corr": cmd_corr, "neuro": cmd_neuro}[a.cmd](a)


if __name__ == "__main__":
    main()
