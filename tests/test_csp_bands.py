"""Три полосовые CSP (4–15, 15–25, 25–30 Гц): форма данных, порядок каналов, эквивалентность реализаций.

Для каждой полосы b: сырой сигнал → причинный полосовой фильтр b → CAR по 13 каналам CSP → окно 250.
1. Онлайн-путь (ковариации EEGFrontEnd.A[:, b] → M_car → features_from_cov) ≡ эталону по сигналу
   (vectorize_windows) — до 1e-6.
2. Если установлен mne: исходный код коллеги (vectorize с объектом mne.CSP из research/csp/bands_src)
   на тех же окнах даёт те же CSP log-var и лог-мощность (асимметрия у коллеги — mean(x²), на полосовом
   сигнале с нулевым средним совпадает с дисперсией до эффектов окна).
3. Размерности: вход (n, 13, 250) → CSP (n, 8, 250) → 8 + 13 + 4 = 25 признаков на полосу, 75 на три.

  python tests/test_csp_bands.py [session.mat]
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from sim import ROOT, default_session, iu

sys.path.insert(0, str(ROOT / "solution"))
from hbci.csp import CSPArtifact, features_from_cov, vectorize_windows  # noqa: E402
from hbci.eeg import EEGFrontEnd, spatial_matrix  # noqa: E402
from hbci.filters import StreamingSOS, butter_sos  # noqa: E402
from test_csp_features import vectorize  # noqa: E402  (исходный код коллеги)

BANDS = ["4-15", "15-25", "25-30"]
SYM = [("c3", "c4"), ("f3", "f4"), ("p3", "p4"), ("c7", "c8")]


def run(path) -> dict:
    s = iu.load_session(path)
    arts = [CSPArtifact.from_dict(joblib.load(ROOT / "research" / "csp" / "bands" / f"csp_{b}Hz_colleague.joblib"))
            for b in BANDS]
    CH = arts[0].channels
    assert all(a.channels == CH for a in arts), "разный порядок каналов у CSP"
    assert all(a.reference == "car" for a in arts), "ожидается CAR"
    names = iu.EEG_CHANNEL_NAMES
    ix = [names.index(c) for c in CH]
    cfg = {"channels": "sensorimotor11", "bands": [[8, 12]], "band_order": 4, "hp_hz": 0.5, "features": "ts",
           "csp": {"enabled": True, "models": [{"prep": a.prep} for a in arts]}}
    n = 40000
    fe = EEGFrontEnd(cfg, 250.0, names, len(s["states"]), 250, 62)
    fe.push(s["eeg"][:n])
    ks = np.arange(20, fe.K, 37)
    z = s["eeg"][:n] - s["eeg"][0]
    z = StreamingSOS(butter_sos(2, 0.5, 250.0, "highpass"))(z)
    out: dict = {"n_windows": int(len(ks)), "channels": CH}
    total = 0
    try:
        import mne  # noqa: F401
        have_mne = True
    except ImportError:
        have_mne = False
    for i, (b, a) in enumerate(zip(BANDS, arts)):
        y = StreamingSOS(butter_sos(4, a.prep, 250.0, "bandpass"))(z)[:, ix]
        y = y - y.mean(1, keepdims=True)                                      # CAR по 13 каналам
        win = np.stack([y[e - 250:e].T for e in fe.ends[ks]])                 # (n, 13, 250)
        ref = vectorize_windows(win, a.filters, CH, SYM)
        M, _ = spatial_matrix(names, CH, "none", set(), interpolate=True)
        M = M - M.mean(0, keepdims=True)
        got = features_from_cov(M @ fe.A[ks, i] @ M.T, a.filters, CH, SYM)
        r = {"input": list(win.shape), "csp_out": [len(ks), a.filters.shape[0], 250], "n_features": int(ref.shape[1]),
             "max_abs_diff_online_vs_signal": float(np.abs(ref - got).max())}
        if have_mne:
            csp = joblib.load(ROOT / "research" / "csp" / "bands_src" / f"csp_{b}Hz.joblib")
            col = vectorize(win, csp, [[p.upper(), q.upper()] for p, q in SYM])
            r["colleague_shape"] = list(col.shape)
            r["colleague_csp_power_diff"] = float(np.abs(col[:, :21] - ref[:, :21]).max())
            r["colleague_asym_diff"] = float(np.abs(col[:, 21:] - ref[:, 21:]).max())
        total += ref.shape[1]
        out[b] = r
    out["total_features_3_bands"] = total
    return out


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else default_session())
    for k, v in r.items():
        print(k, v)
    for b in BANDS:
        assert r[b]["max_abs_diff_online_vs_signal"] < 1e-6, f"{b}: онлайн ≠ сигнал"
        if "colleague_csp_power_diff" in r[b]:
            assert r[b]["colleague_csp_power_diff"] < 1e-6, f"{b}: расхождение с кодом коллеги"
    print("OK: три полосовые CSP — формы, порядок каналов и признаки согласованы")
