"""Бюджеты времени на самой длинной сессии (однопоточно): max predict < 0.05 с, max fit_block < 1.0 с.

  OMP_NUM_THREADS=1 python tests/test_timing.py [session.mat]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys  # noqa: E402

from sim import ROOT, iu, simulate  # noqa: E402


def longest() -> str:
    import json
    p = ROOT / "results" / "cache" / "sessions.json"
    if p.exists():
        info = json.loads(p.read_text())
        return str(ROOT / "data" / "train" / (max(info, key=lambda k: info[k]["n"]) + ".mat"))
    files = sorted((ROOT / "data" / "train").glob("*.mat"))
    return str(max(files, key=lambda f: len(iu.load_session(f)["states"])))


def run(path) -> dict:
    s = iu.load_session(path)
    _, (tp, tf) = simulate(s, timings=True)
    return {"session": s["name"], "predict_max_s": max(tp), "predict_mean_s": sum(tp) / len(tp),
            "fit_max_s": max(tf), "fit_mean_s": sum(tf) / len(tf)}


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else longest())
    print(r)
    assert r["predict_max_s"] < 0.05, "predict слишком медленный"
    assert r["fit_max_s"] < 1.0, "fit_block слишком медленный"
    print("OK: бюджеты времени соблюдены")
