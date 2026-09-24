"""Сила артефакта ФЭС по сегментам (только анализ, НЕ вход модели) → results/cache/fes_segments.csv, fes_strength.csv.

Для каждого сегмента: мощность 81.5–85.5 Гц (среднее по «хорошим» каналам) в окне 0.5–9 с сегмента,
в дБ относительно средней мощности по сегментам покоя той же сессии. fes_strength.csv — медиана по сегментам
воображения сессии. Используется для стратификации проб (< 2 дБ vs > 6 дБ, §5.6) и фигуры controls_scatter.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DATA, FS, segments_of  # noqa: E402


def one(path: str) -> list[dict]:
    import io_utils as iu
    s = iu.load_session(path)
    X = s["eeg"] - np.median(s["eeg"], 0)
    flat = (np.diff(X, axis=0) == 0).mean(0)
    Y = sosfiltfilt(butter(4, [81.5, 85.5], "bandpass", fs=FS, output="sos"), X[:, flat < 0.05], axis=0) ** 2
    segs = segments_of(s["states"], s["blocks"])
    pw = np.array([Y[g["start"] + 125 : g["end"]].mean() for g in segs])
    rest = pw[[g["cls"] == 1 for g in segs]].mean()
    return [{"session": Path(path).stem, "start": g["start"], "end": g["end"], "cls": g["cls"], "block": g["block"],
             "fes_db": 10 * np.log10(p / rest)} for g, p in zip(segs, pw)]


if __name__ == "__main__":
    with Pool(8) as pool:
        rows = [r for rr in pool.map(one, [str(f) for f in sorted(DATA.glob("*.mat"))]) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "fes_segments.csv", index=False, float_format="%.3f")
    im = df[df.cls > 1]
    im.groupby("session")["fes_db"].median().rename("fes_db").to_csv(CACHE / "fes_strength.csv", float_format="%.3f")
    print(im.fes_db.describe().round(2).to_dict())
    print("доля проб воображения < 2 дБ:", round((im.fes_db < 2).mean(), 3), " > 6 дБ:", round((im.fes_db > 6).mean(), 3))
