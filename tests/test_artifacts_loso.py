"""LOSO-артефакты: у модели субъекта S в метаданных train_files нет сессий S; финальная — все субъекты.

  python tests/test_artifacts_loso.py [solution/artifacts]
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

from sim import ROOT


def run(art: Path) -> list[str]:
    errs = []
    loso = art / "loso"
    subs = sorted(p for p in loso.iterdir() if p.is_dir()) if loso.exists() else []
    for d in subs:
        for fname in ("global_eeg.joblib", "csp_filters.joblib"):
            errs += _check(d, fname)
    g = art / "global_eeg.joblib"
    if g.exists():
        subj = {Path(x).stem.split("_")[-1] for x in joblib.load(g)["meta"]["train_files"]}
        print(f"финальная глобальная модель: {len(subj)} субъектов")
    print(f"проверено LOSO-каталогов: {len(subs)}")
    return errs


def _check(d: Path, fname: str) -> list[str]:
    """train_files модели в каталоге субъекта d не содержат его сессий (CSP коллеги без train_files — пропуск)."""
    errs = []
    f = d / fname
    if f.exists():
        files = joblib.load(f)["meta"].get("train_files")
        if not isinstance(files, list):
            return errs
        bad = [x for x in files if Path(x).stem.split("_")[-1] == d.name]
        if bad:
            errs.append(f"{d.name}/{fname}: в обучении есть сессии субъекта: {bad[:3]}")
        if not files:
            errs.append(f"{d.name}/{fname}: пустой train_files")
    return errs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirs = [Path(a) for a in sys.argv[1:]]
    else:  # финальные артефакты + все экспериментальные LOSO-каталоги
        dirs = [ROOT / "solution" / "artifacts"] + sorted(p for p in (ROOT / "results" / "artifacts_exp").glob("*")
                                                          if p.is_dir())
    errs = []
    for art in dirs:
        print(f"== {art.relative_to(ROOT) if art.is_relative_to(ROOT) else art}")
        errs += run(art)
    for e in errs:
        print("ОШИБКА:", e)
    assert not errs, "LOSO-артефакты содержат данные оцениваемого субъекта"
    print("OK: LOSO-артефакты не содержат оцениваемого субъекта")
