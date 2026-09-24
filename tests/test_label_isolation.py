"""Изоляция меток: перестановка States в блоках > b не меняет ответов в окнах блоков ≤ b (b = 3).

Метки сегментов внутри каждого блока > b случайно переставляются (границы сегментов сохраняются,
поэтому набор оцениваемых окон тот же).

  python tests/test_label_isolation.py [session.mat]
"""
from __future__ import annotations

import sys

import numpy as np

from sim import default_session, iu, simulate

B = 3


def permute_after(s: dict, b: int, seed: int = 0) -> dict:
    s = dict(s)
    st = s["states"].copy()
    bl = s["blocks"]
    rng = np.random.default_rng(seed)
    d = np.flatnonzero(np.diff(np.r_[-1, st, -1]) != 0)
    for blk in np.unique(bl[bl > b]):
        segs = [(x, y) for x, y in zip(d[:-1], d[1:]) if st[x] > 0 and bl[x] == blk]
        labs = [int(st[x]) for x, _ in segs]
        perm = rng.permutation(labs)
        while len(labs) > 1 and np.array_equal(perm, labs):
            perm = rng.permutation(labs)
        for (x, y), c in zip(segs, perm):
            st[x:y] = c
    s["states"] = st
    return s


def run(path) -> dict:
    s = iu.load_session(path)
    a = simulate(s)
    sp = permute_after(s, B)
    b = simulate(sp)
    bl = s["blocks"]
    ea = {e: r for e, r in a}
    eb = {e: r for e, r in b}
    early = [e for e in ea if bl[e - 1] <= B]
    diff = sum(ea[e]["y"] != eb[e]["y"] or ea[e].get("y_nirs") != eb[e].get("y_nirs") for e in early if e in eb)
    later = sum(ea[e]["y"] != eb[e]["y"] for e in ea if bl[e - 1] > B and e in eb)
    return {"session": s["name"], "n_early": len(early), "diff_early": diff, "diff_later": later,
            "ok": diff == 0 and later > 0 and len(early) > 0}


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else default_session())
    print(r)
    assert r["diff_early"] == 0, "УТЕЧКА МЕТОК: ответы блоков ≤ 3 зависят от меток будущих блоков"
    assert r["diff_later"] > 0, "перестановка не подействовала — тест некорректен"
    print("OK: метки будущих блоков не влияют на прошлые ответы")
