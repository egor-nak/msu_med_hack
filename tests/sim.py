"""Симулятор порядка событий run.py для тестов утечек (с управляемым разбиением push).

simulate(session, cutter) повторяет run.py: события (predict в конце каждого оцениваемого окна,
fit_block в конце каждого блока; predict раньше fit при равном времени) и push всех новых
отсчётов перед событием. cutter(cur, end) → список точек разреза внутри [cur, end] — позволяет
дробить порции push (тест инвариантности к разбиению).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import classifier as C  # noqa: E402
import io_utils as iu  # noqa: E402

ARTIFACTS = ROOT / "solution" / "artifacts"


def _nirs(s, a, b):
    if s["has_nirs"]:
        return iu.nirs_slice_for_eeg_range(s, a, b)
    n = int(s.get("n_nirs_ch") or 0)
    return np.zeros((0, n)), np.zeros((0, n))


def simulate(s: dict, cutter=None, timings: bool = False, cls=None):
    """→ список (end, predict_dict) и, при timings, (predict_times, fit_times)."""
    cls = cls or C.OnlineClassifier
    clf = cls(iu.session_meta(s))
    if ARTIFACTS.exists():
        clf.load(ARTIFACTS)
    st, bl = s["states"], s["blocks"]
    ev = [(e, 0, None) for _, e, _ in iu.iter_scored_windows(st, bl, s["fs_eeg"])]
    ev += [(int(np.flatnonzero(bl == b)[-1]) + 1, 1, int(b)) for b in np.unique(bl) if b >= 1]
    out, cur = [], 0
    tp, tf = [], []
    for end, kind, b in sorted(ev):
        if end > cur:
            cuts = cutter(cur, end) if cutter else [end]
            a = cur
            for c in cuts:
                if c > a:
                    clf.push(s["eeg"][a:c], _nirs(s, a, c))
                    a = c
            cur = end
        if kind:
            m = (bl == b) & (st > 0)
            t0 = time.perf_counter()
            clf.fit_block(b, {"states": st[m].copy(), "eeg_indices": np.flatnonzero(m), "block": b})
            tf.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            r = clf.predict()
            tp.append(time.perf_counter() - t0)
            out.append((end, r))
    return (out, (tp, tf)) if timings else out


def default_session() -> Path:
    return sorted((ROOT / "data" / "train").glob("*.mat"))[0]
