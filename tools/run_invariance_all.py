"""§5.8: tests/test_chunk_invariance на всех 56 сессиях (текущая конфигурация) → results/tests.json.

  python tools/run_invariance_all.py [-j 8]
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def job(path: str) -> dict:
    import test_chunk_invariance as T
    return T.run(path)


if __name__ == "__main__":
    j = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 8
    files = [str(f) for f in sorted((ROOT / "data" / "train").glob("*.mat"))]
    t0 = time.time()
    with Pool(j) as pool:
        R = pool.map(job, files, chunksize=1)
    out = {"test": "chunk_invariance (порции по 62 и по 1 отсчёту)", "config": os.environ.get("OC_CFG") or "artifacts/config.json",
           "n_sessions": len(R), "all_ok": all(r["ok"] for r in R), "n_windows": sum(r["n"] for r in R),
           "total_diff": sum(r["diff_y_62"] + r["diff_nirs_62"] + r["diff_y_1"] + r["diff_nirs_1"] for r in R),
           "seconds": round(time.time() - t0, 1), "sessions": R}
    (ROOT / "results" / "tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print({k: v for k, v in out.items() if k != "sessions"})
