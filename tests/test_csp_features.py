"""CSP/мощность/асимметрия: эквивалентность реализаций и совместимость с обученной CSP.

1. Онлайн-путь (ковариации окон из EEGFrontEnd → hbci.csp.features_from_cov) совпадает с эталонным
   векторизатором по сигналу (hbci.csp.vectorize_windows) — CSP log-var и лог-мощность до 1e-9.
2. Если установлен mne: исходный код коллеги (vectorize, ниже без изменений) с объектом
   csp_model.joblib даёт те же CSP log-var и лог-мощность; асимметрия отличается — у коллеги
   P = mean(x²) без вычитания среднего, и на сырых отсчётах с постоянной составляющей она описывает DC.

  python tests/test_csp_features.py [session.mat]
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from sim import ROOT, default_session, iu

sys.path.insert(0, str(ROOT / "solution"))
from hbci.csp import CSPArtifact, features_from_cov, vectorize_windows  # noqa: E402
from hbci.eeg import EEGFrontEnd  # noqa: E402

CSP_NP = ROOT / "research" / "csp" / "csp_filters_colleague.joblib"
CSP_MNE = ROOT / "research" / "csp" / "csp_model.joblib"
SYM = [("c3", "c4"), ("f3", "f4"), ("p3", "p4"), ("c7", "c8")]


def vectorize(data, csp, symmetry=None, return_powers=True):  # исходный код коллеги (эталон, без правок)
    chan_dict = {'C3': 0, 'Cz': 1, 'C4': 2, 'F3': 3, 'Fz': 4, 'F4': 5, 'P3': 6, 'Pz': 7, 'P4': 8, 'C7': 9, 'C8': 10}
    csp_vectors = np.log(np.var(csp.transform(data), axis=-1))
    if return_powers:
        power_vectors = np.log(np.var(data, axis=-1, ddof=0))
        csp_vectors = np.concatenate([csp_vectors, power_vectors], axis=1)
    if not (symmetry == None):  # noqa: E711
        for sym_pair in symmetry:
            P_left = np.mean(data[:, chan_dict[sym_pair[0]], :] ** 2, axis=-1)
            P_right = np.mean(data[:, chan_dict[sym_pair[1]], :] ** 2, axis=-1)
            denominator = P_left + P_right
            asymmetry = np.divide(P_left - P_right, denominator, out=np.zeros_like(denominator),
                                  where=denominator > 0).reshape(-1, 1)
            csp_vectors = np.concatenate([csp_vectors, asymmetry], axis=1)
    return csp_vectors


def run(path) -> dict:
    s = iu.load_session(path)
    art = CSPArtifact.from_dict(joblib.load(CSP_NP))
    ix = [iu.EEG_CHANNEL_NAMES.index(c) for c in art.channels]
    cfg = {"channels": "sensorimotor11", "bands": [[8, 12]], "band_order": 4, "hp_hz": 0.5, "features": "ts",
           "csp": {"enabled": True, "prep": "raw"}}
    fe = EEGFrontEnd(cfg, 250.0, iu.EEG_CHANNEL_NAMES, len(s["states"]), 250, 62)
    fe.push(s["eeg"][:40000])
    ks = np.arange(20, fe.K, 37)
    x0 = s["eeg"][0]
    win = np.stack([(s["eeg"][e - 250:e] - x0)[:, ix].T for e in fe.ends[ks]])      # (n, 11, 250)
    ref = vectorize_windows(win, art.filters, art.channels, SYM)
    got = features_from_cov(fe.A[ks][:, ix][:, :, ix], art.filters, art.channels, SYM)
    out = {"n": len(ks), "dim": ref.shape[1], "max_abs_diff_cov_vs_signal": float(np.abs(ref - got).max())}
    try:
        import mne  # noqa: F401
        csp = joblib.load(CSP_MNE)
        col = vectorize(win, csp, [[a.upper(), b.upper()] for a, b in SYM])
        out["colleague_csp_power_diff"] = float(np.abs(col[:, :19] - ref[:, :19]).max())
        out["colleague_asym_diff"] = float(np.abs(col[:, 19:] - ref[:, 19:]).max())
        out["colleague_asym_range"] = [float(col[:, 19:].min()), float(col[:, 19:].max())]
        out["demeaned_asym_range"] = [float(ref[:, 19:].min()), float(ref[:, 19:].max())]
    except ImportError:
        out["colleague"] = "mne не установлен — сравнение с исходным кодом пропущено"
    return out


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else default_session())
    print(r)
    assert r["max_abs_diff_cov_vs_signal"] < 1e-6, "онлайн-признаки (по ковариации) ≠ признакам по сигналу"
    if "colleague_csp_power_diff" in r:
        assert r["colleague_csp_power_diff"] < 1e-6, "CSP/лог-мощность не совпадают с исходным кодом коллеги"
    print("OK: CSP-признаки по ковариации совпадают с векторизатором по сигналу")
