"""Прогон сетки конфигураций через замороженные run.py/score.py (логика tools/run_all.one).

  python tools/run_experiments.py --exp-id E04_riemann --grid grids/e04.json -j 8
  python tools/run_experiments.py --exp-id E01_b1fix --cfg '{"preset":"b1fix"}' --name b1fix
  python tools/run_experiments.py --exp-id E08_transfer --grid g.json --loso solution/artifacts/loso

Для каждой конфигурации выставляется OC_CFG (JSON), HBCI_DUMP_DIR (вероятности по окнам)
и, при --loso, HBCI_LOSO_DIR. Результат: results/experiments/<exp_id>/<name>/
(per_session/, runs/, dumps/, metrics.json, summary.csv, config.json) + tools/analyze.py.
Один рабочий процесс прогоняет все конфигурации на своей сессии (сессия читается один раз).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "tools"))
EXP = ROOT / "results" / "experiments"


def _worker(job: tuple) -> list[tuple[str, str, dict]]:
    path, cfgs, artifacts, loso = job
    import io_utils as iu
    import run_all as RA
    orig = iu.load_session
    cache: dict = {}

    def cached(p):
        k = str(p)
        if k not in cache:
            cache[k] = orig(p)
        return cache[k]

    iu.load_session = cached
    import classifier as CM
    orig_cls = CM.OnlineClassifier
    out = []
    try:
        for name, cj, od in cfgs:
            os.environ["OC_CFG"] = cj
            os.environ["HBCI_DUMP_DIR"] = str(Path(od) / "dumps")
            impl = json.loads(cj).get("_impl")
            if impl:  # только эксперименты: исходный MVP / B1-fix в legacy/
                import importlib
                CM.OnlineClassifier = importlib.import_module(f"legacy.{impl}").OnlineClassifier
            else:
                CM.OnlineClassifier = orig_cls
            lo = json.loads(cj).get("_loso") or loso  # каталог LOSO-артефактов на уровне конфигурации
            if lo:
                os.environ["HBCI_LOSO_DIR"] = str((ROOT / lo).resolve()) if not os.path.isabs(lo) else lo
            else:
                os.environ.pop("HBCI_LOSO_DIR", None)
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                stem, res = RA.one((path, od, artifacts))
            if "error" in res:
                print(f"FAIL {name} {stem}: {res['error']}", flush=True)
            out.append((name, stem, res))
    finally:
        iu.load_session = orig
        CM.OnlineClassifier = orig_cls
    return out


def load_grid(a) -> dict[str, dict]:
    if a.grid:
        g = json.loads(Path(a.grid).read_text())
        if isinstance(g, list):
            g = {x["name"]: x["cfg"] for x in g}
        return g
    if a.config:
        return {a.name or Path(a.config).stem: json.loads(Path(a.config).read_text())}
    if a.cfg is not None:
        return {a.name or "cfg": json.loads(a.cfg)}
    raise SystemExit("нужен --grid, --config или --cfg")


def run_grid(exp_id: str, grid: dict[str, dict], data: str = str(ROOT / "data" / "train"), j: int = 8,
             loso: str | None = None, limit: int = 0, sessions: list[str] | None = None,
             skip_existing: bool = False, analyze: bool = True, only: list[str] | None = None) -> dict[str, dict]:
    import run_all as RA
    files = sorted(Path(data).glob("*.mat"))
    if sessions:
        files = [f for f in files if f.stem in sessions]
    if limit:
        files = files[:: max(1, len(files) // limit)][:limit]
    todo = []
    for name, cfg in grid.items():
        if only and name not in only:
            continue
        od = EXP / exp_id / name
        if skip_existing and (od / "metrics.json").exists():
            continue
        od.mkdir(parents=True, exist_ok=True)
        (od / "config.json").write_text(json.dumps({"cfg": cfg, "loso": cfg.get("_loso") or loso, "n_sessions": len(files)},
                                                   ensure_ascii=False, indent=2))
        todo.append((name, json.dumps(cfg), str(od)))
    if todo:
        t0 = time.time()
        jobs = [(str(f), todo, str(ROOT / "solution" / "artifacts"), loso) for f in files]
        if j == 1:
            list(map(_worker, jobs))
        else:
            with Pool(j) as pool:
                pool.map(_worker, jobs, chunksize=1)
        print(f"[{exp_id}] {len(todo)} конф. × {len(files)} сессий: {time.time() - t0:.0f} с", flush=True)
    res = {}
    for name in grid:
        if only and name not in only:
            continue
        od = EXP / exp_id / name
        if not (od / "per_session").exists():
            continue
        agg = RA.aggregate(od, verbose=False)
        res[name] = agg
        h, n = agg["hybrid"], agg.get("nirs_only") or {}
        print(f"  {name:28s} F={h['min_f1']:.4f} P={h['macro_recall']:.4f} NF={n.get('min_f1', 0):.4f} "
              f"EEG_F={agg.get('eeg_only_info', {}).get('min_f1', 0):.4f} pen={agg['timings']['penalty_total']} "
              f"score={agg['score']['total']:.2f}", flush=True)
        if analyze:
            import analyze as AN
            AN.analyze_run(od, quiet=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--grid")
    ap.add_argument("--config")
    ap.add_argument("--cfg")
    ap.add_argument("--name")
    ap.add_argument("--data", default=str(ROOT / "data" / "train"))
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--loso", default=None, help="каталог LOSO-артефактов (HBCI_LOSO_DIR)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    a = ap.parse_args()
    loso = str(Path(a.loso).resolve()) if a.loso else None
    run_grid(a.exp_id, load_grid(a), a.data, a.j, loso, a.limit, None, a.skip_existing, not a.no_analyze, a.only)


if __name__ == "__main__":
    main()
