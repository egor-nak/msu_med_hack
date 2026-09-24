"""Особые сессии проходят через run.py без исключений: без NIRS (y_nirs отсутствует) и NIRS 3.90625 Гц.

  python tests/test_edge_sessions.py [data_dir]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

from sim import ROOT

NO_NIRS = ["2025.12.18.12.09.22_N03", "2025.12.18.15.02.14_N02"]
LOW_FS = ["2025.12.09.12.52.43_N01", "2026.02.19.13.23.35_N13"]


def run(data: Path) -> list[dict]:
    import run as R
    out = []
    with tempfile.TemporaryDirectory() as td:
        for stem in NO_NIRS + LOW_FS:
            od = Path(td) / stem
            R.run_session(data / f"{stem}.mat", od, ROOT / "solution" / "artifacts")
            p = pd.read_csv(od / "predictions.csv")
            has = "y_nirs" in p and p["y_nirs"].notna().any()
            ok = (not has) if stem in NO_NIRS else (has and p["y_nirs"].isin([1, 2, 3]).all())
            out.append({"session": stem, "n": len(p), "y_nirs": bool(has), "ok": bool(ok)})
    return out


if __name__ == "__main__":
    data = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else ROOT / "data" / "train"
    res = run(data)
    for r in res:
        print(r)
    assert all(r["ok"] for r in res), "особые сессии обработаны некорректно"
    print("OK: сессии без NIRS и с NIRS 3.9 Гц проходят")
