"""Nested-оценка выбора конфигурации (§6): внешний цикл — 7 фолдов по 2 субъекта.

Во внутреннем цикле из ≤ 20 кандидатов (LOSO-прогоны) выбирается конфигурация с максимальным баллом
60·F + 30·P + 5 + 15·F_NIRS на сессиях остальных 12 субъектов; отчётная метрика — на сессиях отложенных 2.
Упрощение: LOSO-артефакты кандидата для субъекта S обучены без S, но с его «напарником» по фолду
(переобучать глобальные модели для каждого фолда слишком дорого); выбор гиперпараметров при этом честный.

  python tools/nested.py --candidates tools/grids/nested_candidates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, subject_of  # noqa: E402

EXP = ROOT / "results" / "experiments"


def per_session(run: str) -> pd.DataFrame:
    rows = []
    for p in sorted((EXP / run / "per_session").glob("*.json")):
        r = json.loads(p.read_text())
        n = r.get("nirs_only") or {}
        rows.append({"session": p.stem, "subject": subject_of(p.stem), "F": r["hybrid"]["min_f1"],
                     "P": r["hybrid"]["macro_recall"], "NF": n.get("min_f1", np.nan)})
    return pd.DataFrame(rows).set_index("session")


def score(d: pd.DataFrame) -> float:
    nf = d["NF"].mean()
    return 60 * d["F"].mean() + 30 * d["P"].mean() + (5 + 15 * nf if np.isfinite(nf) else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    a = ap.parse_args()
    cands: list[str] = json.loads(Path(a.candidates).read_text())
    assert len(cands) <= 20, "не больше 20 кандидатов"
    D = {c: per_session(c) for c in cands}
    subs = sorted(D[cands[0]].subject.unique())
    rng = np.random.default_rng(0)
    order = list(rng.permutation(subs))
    folds = [order[i::7] for i in range(7)]
    picked, held = [], []
    for f in folds:
        inner = {c: score(d[~d.subject.isin(f)]) for c, d in D.items()}
        best = max(inner, key=inner.get)
        picked.append({"fold": f, "choice": best, "inner_score": round(inner[best], 3)})
        held.append(D[best][D[best].subject.isin(f)].assign(choice=best))
    H = pd.concat(held)
    full = {c: score(d) for c, d in D.items()}
    winner = max(full, key=full.get)
    out = {"n_candidates": len(cands), "folds": picked,
           "nested": {"F": H.F.mean(), "P": H.P.mean(), "NF": H.NF.mean(), "score": score(H)},
           "winner_on_all": winner, "winner_loso": {"F": D[winner].F.mean(), "P": D[winner].P.mean(),
                                                    "NF": D[winner].NF.mean(), "score": full[winner]},
           "optimism": full[winner] - score(H)}
    out_p = EXP / "E12_final_loso" / "nested.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
