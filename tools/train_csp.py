"""Офлайн-обучение CSP по рецепту коллеги (mne) — для честной оценки и как место, где CSP должна обучаться.

  python tools/train_csp.py --prep raw   --out results/artifacts_exp/csp_raw        [--mode loso|all]
  python tools/train_csp.py --prep 4,30  --out results/artifacts_exp/csp_bp4-30

Рецепт как у csp_model.joblib: mne.decoding.CSP(n_components=8, reg='ledoit_wolf', cov_est='epoch',
component_order='mutual_info', transform_into='csp_space'); классы 1/2/3; каналы chan_dict
(c3 cz c4 f3 fz f4 p3 pz p4 c7 c8). Данные — окна 1 с (250 отсчётов) целиком внутри размеченных
сегментов всех блоков, с шагом stride точек сетки (по умолчанию 4 → непересекающиеся окна).
Вход CSP строится ТЕМИ ЖЕ причинными фильтрами, что онлайн (hbci.filters): "raw" — x − x0;
[lo, hi] — ВЧ 0.5 Гц + полосовой Баттерворт 4-го порядка. Метки нужны только здесь, офлайн.
--mode loso: для каждого субъекта S — CSP без сессий S → <out>/loso/S/csp_filters.joblib
(+ копия global_eeg.joblib из --global-from для совместных LOSO-экспериментов).
--mode all: одна CSP на всех сессиях → <out>/csp_filters.joblib.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "tools"))
CHAN = ["c3", "cz", "c4", "f3", "fz", "f4", "p3", "pz", "p4", "c7", "c8"]
CHAN13 = CHAN + ["p7", "p8"]   # набор полосовых CSP коллеги (csp_4-15/15-25/25-30Hz.joblib)
CACHE = ROOT / "results" / "cache" / "csp_windows"


def session_windows(path: str, prep, stride: int, chans: list[str] = CHAN, car: bool = False) -> tuple[np.ndarray, np.ndarray]:
    import io_utils as iu
    from common import segments_of
    from hbci.filters import StreamingSOS, butter_sos
    s = iu.load_session(path)
    X = s["eeg"].astype(np.float64)
    z = X - X[0]
    if prep != "raw":
        z = StreamingSOS(butter_sos(2, 0.5, 250.0, "highpass"))(z)
        z = StreamingSOS(butter_sos(4, list(prep), 250.0, "bandpass"))(z)
    ix = [iu.EEG_CHANNEL_NAMES.index(c) for c in chans]
    z = z[:, ix]
    if car:
        z = z - z.mean(1, keepdims=True)   # общий средний референс по каналам CSP (как у полосовых CSP коллеги)
    W, y = [], []
    for g in segments_of(s["states"], s["blocks"]):
        k0 = -(-g["start"] // 62)
        for k in range(k0, (g["end"] - 250) // 62 + 1, stride):
            e = 250 + 62 * k
            W.append(z[e - 250:e].T)
            y.append(g["cls"])
    return np.asarray(W, dtype=np.float32), np.asarray(y, dtype=np.int8)


def cached(path: str, prep, stride: int, chans: list[str] = CHAN, car: bool = False) -> tuple[np.ndarray, np.ndarray]:
    tag = ("raw" if prep == "raw" else f"bp{prep[0]:g}-{prep[1]:g}") + (f"_{len(chans)}ch" if chans != CHAN else "") \
        + ("_car" if car else "")
    p = CACHE / f"{tag}_s{stride}" / (Path(path).stem + ".npz")
    if p.exists():
        z = np.load(p)
        return z["W"], z["y"]
    W, y = session_windows(path, prep, stride, chans, car)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, W=W, y=y)
    return W, y


def fit_csp(W: np.ndarray, y: np.ndarray, files: list[str], prep, chans: list[str] = CHAN, car: bool = False) -> dict:
    import mne
    kw = {}
    if car:  # после CAR ранг данных n−1: как у CSP коллеги — info + rank={'eeg': n−1}
        kw = {"info": mne.create_info([c.upper() for c in chans], 250.0, "eeg"), "rank": {"eeg": len(chans) - 1}}
    csp = mne.decoding.CSP(n_components=8, reg="ledoit_wolf", cov_est="epoch", component_order="mutual_info",
                           transform_into="csp_space", log=None, **kw)
    mne.set_log_level("ERROR")
    csp.fit(W.astype(np.float64), y.astype(int if car else float))
    return {"filters": csp.filters_[:8].copy(), "channels": chans, "prep": prep, "reference": "car" if car else "none",
            "meta": {"type": "mne.decoding.CSP (tools/train_csp.py)", "mne_version": mne.__version__,
                     "params": csp.get_params(), "train_files": [Path(f).name for f in files],
                     "n_windows": int(len(y)), "patterns": csp.patterns_[:8].tolist()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="raw", help='"raw" или "lo,hi"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["loso", "all"], default="loso")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--global-from", default=str(ROOT / "results" / "artifacts_exp" / "g_sm11_avg5"))
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--chans13", action="store_true", help="13 каналов (+p7, p8) и CAR — как полосовые CSP коллеги")
    ap.add_argument("--name", default="csp_filters.joblib", help="имя файла артефакта")
    a = ap.parse_args()
    prep = "raw" if a.prep == "raw" else [float(x) for x in a.prep.split(",")]
    files = [str(f) for f in sorted((ROOT / "data" / "train").glob("*.mat"))]
    chans, car = (CHAN13, True) if a.chans13 else (CHAN, False)
    with Pool(a.j) as pool:
        data = pool.starmap(cached, [(f, prep, a.stride, chans, car) for f in files])
    subj = np.array([Path(f).stem.split("_")[-1] for f in files])
    out = Path(a.out)
    if a.mode == "all":
        W = np.concatenate([d[0] for d in data]); y = np.concatenate([d[1] for d in data])
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit_csp(W, y, files, prep, chans, car), out / a.name)
        print(f"→ {out / a.name} ({len(y)} окон)")
        return
    for s in sorted(set(subj)):
        tr = [i for i in range(len(files)) if subj[i] != s]
        W = np.concatenate([data[i][0] for i in tr]); y = np.concatenate([data[i][1] for i in tr])
        d = out / "loso" / s
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit_csp(W, y, [files[i] for i in tr], prep, chans, car), d / a.name)
        g = Path(a.global_from) / "loso" / s / "global_eeg.joblib"
        if g.exists():
            shutil.copy(g, d / "global_eeg.joblib")
        print(f"  LOSO {s}: {len(tr)} сессий, {len(y)} окон → {d}", flush=True)


if __name__ == "__main__":
    main()
