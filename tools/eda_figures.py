"""Офлайн-EDA для фигур 1–5 (§7): ERD/ERS, топографии, артефакт ФЭС, ГЭОГ, NIRS.

Это анализ, не классификатор: здесь допустимы двунаправленные фильтры (sosfiltfilt) и метки.
Агрегаты по сессиям кэшируются в results/cache/eda.npz.

  python tools/eda_figures.py            # посчитать (если нет кэша) и нарисовать фигуры 1–5
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram, welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATA, FS, ROOT, segments_of  # noqa: E402

NAMES = ["c4", "rpa", "f8", "p8", "f4", "p4", "fp2", "o2", "cz", "pz", "fz", "o1", "fp1", "p3", "f3", "p7", "f7",
         "lpa", "c3", "c7", "c8"]
CLEAN = [n for n in NAMES if n not in ("fp1", "fp2", "f7", "f8", "lpa", "rpa", "o1", "o2")]
I = {n: i for i, n in enumerate(NAMES)}
PRE, POST = int(1.5 * FS), int(9.0 * FS)
TF_CH = ["c3", "cz", "c4"]


def _bad(X: np.ndarray) -> np.ndarray:
    v = np.log(np.var(sosfiltfilt(butter(4, [4, 30], "bandpass", fs=FS, output="sos"), X[::1], axis=0), 0) + 1e-30)
    d = v - np.median(v)
    flat = (np.diff(X, axis=0) == 0).mean(0)
    return (d > np.log(8)) | (d < -np.log(400)) | (flat > 0.05)


def one_session(path: str) -> dict:
    import io_utils as iu
    s = iu.load_session(path)
    X = s["eeg"] - np.median(s["eeg"], 0)
    bad = _bad(X)
    hp = butter(2, 0.5, "highpass", fs=FS, output="sos")
    Xh = sosfiltfilt(hp, X, axis=0)
    clean = [I[n] for n in CLEAN if not bad[I[n]]]
    Xc = Xh - Xh[:, clean].mean(1, keepdims=True)
    segs = [g for g in segments_of(s["states"], s["blocks"]) if g["start"] >= PRE and g["start"] + POST <= len(X)]
    cls = np.array([g["cls"] for g in segs])
    st = np.array([g["start"] for g in segs])
    out: dict = {"name": s["name"], "bad": bad}
    # --- 1. TFR на C3/Cz/C4 (STFT 0.5 с), дБ относительно среднего покоя
    tf = []
    for g in st:
        f, t, P = spectrogram(Xc[g - PRE : g + POST, [I[c] for c in TF_CH]].T, fs=FS, nperseg=125, noverlap=94)
        tf.append(P)
    tf = np.array(tf)  # (n, ch, f, t)
    fm = (f >= 4) & (f <= 40)
    tf = tf[:, :, fm]
    t = t - PRE / FS
    base = tf[cls == 1][..., (t >= 0.5)].mean(axis=(0, 3), keepdims=True)[0]
    out["tfr"] = np.stack([10 * np.log10(tf[cls == c].mean(0) / base) for c in (1, 2, 3)])
    out["tfr_f"], out["tfr_t"] = f[fm], t
    # --- 2. топографии: μ 8–13 и β 15–30, окно 0.5–9 с, дБ по классам
    tp = {}
    for name, band in (("mu", (8, 13)), ("beta", (15, 30))):
        Y = sosfiltfilt(butter(4, band, "bandpass", fs=FS, output="sos"), Xc, axis=0) ** 2
        pw = np.array([Y[g + 125 : g + POST].mean(0) for g in st])
        L = 10 * np.log10(pw + 1e-30)
        L[:, bad] = np.nan
        tp[name] = np.stack([np.nanmean(L[cls == c], 0) for c in (1, 2, 3)])
    out["topo_mu"], out["topo_beta"] = tp["mu"], tp["beta"]
    # --- 3. ФЭС: спектр воображение/покой 1–125 Гц и ход мощности 81.5–85.5 Гц в сегменте
    psd = []
    for g in st:
        fw, Pw = welch(Xh[g + 125 : g + POST].T, fs=FS, nperseg=500)
        psd.append(Pw)
    psd = np.array(psd)  # (n, ch, f)
    ok = ~bad
    ratio = 10 * np.log10(psd[cls > 1][:, ok].mean(0) / psd[cls == 1][:, ok].mean(0))
    out["fes_f"], out["fes_ratio"] = fw, np.median(ratio, 0)
    Y = sosfiltfilt(butter(4, [81.5, 85.5], "bandpass", fs=FS, output="sos"), Xh[:, ok], axis=0) ** 2
    w = int(0.5 * FS)
    tc = np.array([[Y[g - PRE + k : g - PRE + k + w].mean() for k in range(0, PRE + POST - w + 1, w // 2)] for g in st])
    ref = tc[cls == 1].mean()
    out["fes_tc"] = np.stack([10 * np.log10(tc[cls == c].mean(0) / ref) for c in (1, 2, 3)])
    out["fes_tc_t"] = (np.arange(0, PRE + POST - w + 1, w // 2) + w / 2 - PRE) / FS
    # --- 4. ГЭОГ f7−f8, 0.1–3 Гц, базлайн [−1.2, 0] с, в единицах SD сессии
    h = sosfiltfilt(butter(2, [0.1, 3], "bandpass", fs=FS, output="sos"), X[:, I["f7"]] - X[:, I["f8"]])
    h = h / (np.std(h) + 1e-12)
    ep = np.array([h[g - PRE : g + POST] - h[g - 300 : g].mean() for g in st])
    out["heog"] = np.stack([ep[cls == c].mean(0) for c in (1, 2, 3)])
    out["heog_t"] = (np.arange(-PRE, POST)) / FS
    # --- 5. NIRS: HbO/HbR по классам и полушариям (только 12.5 Гц), базлайн [−2, 0] с
    if s["has_nirs"] and abs(s["fs_nirs"] - 12.5) < 1e-6:
        fsn = s["fs_nirs"]
        sos = butter(2, [0.01, 0.4], "bandpass", fs=fsn, output="sos")
        res = []
        for key in ("nirs_hbo", "nirs_hbr"):
            Z = sosfiltfilt(sos, s[key], axis=0)
            Z = Z / (Z.std(0) + 1e-30)
            hemi = np.stack([Z[:, 8:16].mean(1), Z[:, 0:8].mean(1)], 1)  # левое, правое
            pre, post = int(2 * fsn), int(15 * fsn)
            idx = np.searchsorted(s["nirs_frame"], st + 1)
            eps = np.array([hemi[i - pre : i + post] - hemi[i - pre : i].mean(0) for i in idx
                            if i - pre >= 0 and i + post <= len(hemi)])
            cc = np.array([c for c, i in zip(cls, idx) if i - pre >= 0 and i + post <= len(hemi)])
            res.append(np.stack([eps[cc == c].mean(0) for c in (1, 2, 3)]))
        out["nirs"] = np.stack(res)  # (Hb, class, t, hemi)
        out["nirs_t"] = np.arange(-int(2 * fsn), int(15 * fsn)) / fsn
    return out


def compute(refresh: bool = False) -> dict:
    p = CACHE / "eda.npz"
    if p.exists() and not refresh:
        return dict(np.load(p, allow_pickle=True))
    files = [str(f) for f in sorted(DATA.glob("*.mat"))]
    with Pool(8) as pool:
        R = pool.map(one_session, files)
    agg = {k: np.stack([r[k] for r in R]) for k in ("tfr", "topo_mu", "topo_beta", "fes_ratio", "fes_tc", "heog")}
    for k in ("tfr_f", "tfr_t", "fes_f", "fes_tc_t", "heog_t"):
        agg[k] = R[0][k]
    N = [r for r in R if "nirs" in r]
    agg["nirs"] = np.stack([r["nirs"] for r in N])
    agg["nirs_t"] = N[0]["nirs_t"]
    agg["names"] = np.array([r["name"] for r in R])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(p, **agg)
    return agg


if __name__ == "__main__":
    import make_figures as MF
    MF.eda_figures(compute("--refresh" in sys.argv))
    print("готово:", ROOT / "results" / "figures")
