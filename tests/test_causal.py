"""Проверка причинности: подмена данных ПОСЛЕ момента T не меняет ни одного ответа ДО T.

  python tests/test_causal.py [session.mat]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import classifier as C  # noqa: E402
import io_utils as iu  # noqa: E402


def simulate(s, T=None):
    """Упрощённый повтор порядка событий run.py; если T задан — после T подставляем шум."""
    s = dict(s)
    if T is not None:
        rng = np.random.default_rng(1)
        s["eeg"] = s["eeg"].copy(); s["eeg"][T:] = rng.normal(0, 1e8, s["eeg"][T:].shape)
        if s["has_nirs"]:
            m = s["nirs_frame"] > T
            for k in ("nirs_hbo", "nirs_hbr"):
                s[k] = s[k].copy(); s[k][m] = rng.normal(0, 1e-3, (m.sum(), s[k].shape[1]))
    clf = C.OnlineClassifier(iu.session_meta(s)); clf.load(ROOT / "solution" / "artifacts")
    st, bl = s["states"], s["blocks"]
    ev = [(e, 0, None) for _, e, _ in iu.iter_scored_windows(st, bl, s["fs_eeg"])]
    ev += [(int(np.flatnonzero(bl == b)[-1]) + 1, 1, b) for b in np.unique(bl)]
    out, cur = [], 0
    for end, kind, b in sorted(ev):
        if end > cur:
            nc = iu.nirs_slice_for_eeg_range(s, cur, end) if s["has_nirs"] else (np.zeros((0, 0)),) * 2
            clf.push(s["eeg"][cur:end], nc); cur = end
        if kind:
            m = (bl == b) & (st > 0)
            clf.fit_block(int(b), {"states": st[m], "eeg_indices": np.flatnonzero(m), "block": int(b)})
        else:
            out.append((end, clf.predict()))
    return out


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else sorted((ROOT / "data" / "train").glob("*.mat"))[0]
    s = iu.load_session(f)
    T = int(0.6 * len(s["states"]))
    a, b = simulate(s), simulate(s, T)
    before = [(x, y) for x, y in zip(a, b) if x[0] <= T]
    bad = [x[0][0] for x in before if x[0][1] != x[1][1]]
    after_diff = sum(x[1] != y[1] for x, y in zip(a, b) if x[0] > T)
    print(f"{Path(f).name}: окон до T={len(before)}, расхождений до T={len(bad)}; после T изменилось {after_diff} ответов")
    assert not bad, f"УТЕЧКА: ответы до T зависят от будущего: {bad[:5]}"
    assert after_diff > 0, "подмена не подействовала — тест некорректен"
    print("OK: решение причинно")
