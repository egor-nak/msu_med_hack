"""Онлайн-симуляция по всем сессиям через замороженные run.py/score.py.

  python tools/run_all.py --data data/train --out results/metrics -j 4
Результат: results/metrics/per_session/<stem>.json, runs/<stem>/*.csv, metrics.json, summary.csv
"""
from __future__ import annotations

import argparse, json, os, sys, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # без этого параллельные прогоны раздувают тайминги
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))


def one(args):
    path, out, artifacts = args
    import run as R, score as S  # noqa
    stem = Path(path).stem
    rd = Path(out) / "runs" / stem
    t0 = time.time()
    try:
        R.run_session(Path(path), rd, Path(artifacts) if artifacts else None)
    except SystemExit as e:
        return stem, {"error": f"run.py exit {e.code}"}
    res = S.score_files(rd / "predictions.csv", rd / "timings.csv", session_path=Path(path))
    # доп. метрика ЭЭГ-only (в балл не входит, для сравнения модальностей)
    import pandas as pd
    p = pd.read_csv(rd / "predictions.csv")
    if "y_eeg" in p:
        res["eeg_only"] = S.metrics_block(S.confusion_from_preds(p.true_y.values, p.y_eeg.values))
    res["wall_s"] = round(time.time() - t0, 1)
    pj = Path(out) / "per_session" / f"{stem}.json"
    pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return stem, res


def aggregate(out: Path, verbose: bool = True) -> dict:
    """Агрегат по per_session/*.json → metrics.json и summary.csv (невзвешенное среднее по сессиям)."""
    import score as S
    results = [(p.stem, json.loads(p.read_text(encoding="utf-8")))
               for p in sorted(Path(out, "per_session").glob("*.json"))]
    ok = [r for _, r in results if "error" not in r]
    for s, r in results:
        if "error" in r:
            print("FAIL", s, r["error"])
    agg = S.aggregate_session_metrics(ok)
    eeg = [r["eeg_only"] for r in ok if "eeg_only" in r]
    if eeg:
        agg["eeg_only_info"] = {k: sum(e[k] for e in eeg) / len(eeg) for k in ("min_f1", "macro_recall")}
    Path(out, "metrics.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = ["session,min_f1,macro_recall,nirs_min_f1,eeg_min_f1,predict_max_s,fit_max_s,windows_match"]
    for s, r in results:
        if "error" in r:
            continue
        n = r.get("nirs_only") or {}
        rows.append(",".join(map(str, [s, round(r["hybrid"]["min_f1"], 4), round(r["hybrid"]["macro_recall"], 4),
                                        round(n.get("min_f1", float("nan")), 4),
                                        round(r.get("eeg_only", {}).get("min_f1", float("nan")), 4),
                                        r["timings"]["predict_max_s"], r["timings"]["fit_max_s"], r.get("windows_match")])))
    Path(out, "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    h, n = agg["hybrid"], agg.get("nirs_only") or {}
    if verbose:
        print(f"sessions={len(ok)}  F_global={h['min_f1']:.4f}  P_global={h['macro_recall']:.4f}  "
              f"NIRS_F={n.get('min_f1', 0):.4f}  EEG_F={agg.get('eeg_only_info', {}).get('min_f1', 0):.4f}  "
              f"pen={agg['timings']['penalty_total']}  SCORE(no design)={agg['score']['total']:.2f}")

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train"))
    ap.add_argument("--out", default=str(ROOT / "results" / "metrics"))
    ap.add_argument("--artifacts", default=str(ROOT / "solution" / "artifacts"))
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="k/n: прогнать только k-ю из n частей (агрегат — по всем json в per_session)")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    import score as S
    files = sorted(Path(a.data).glob("*.mat"))
    if a.limit:
        files = files[:: max(1, len(files) // a.limit)][: a.limit]
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        files = files[k::n]
    if not a.aggregate_only:
        jobs = [(str(f), a.out, a.artifacts) for f in files]
        if a.j == 1:
            list(map(one, jobs))
        else:
            with Pool(a.j) as pool:
                pool.map(one, jobs)
        if a.shard:
            print(f"shard {a.shard}: {len(files)} сессий готово"); return
    aggregate(Path(a.out))


if __name__ == "__main__":
    main()
