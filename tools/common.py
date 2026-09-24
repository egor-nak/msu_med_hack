"""Общие помощники для офлайн-скриптов (анализ, эксперименты, фигуры).

Метки и границы сегментов здесь используются ТОЛЬКО для анализа/оценки, не в классификаторе.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
DATA = ROOT / "data" / "train"
CACHE = ROOT / "results" / "cache"
FS = 250.0


def subject_of(stem: str) -> str:
    return stem.split("_")[-1]


def segments_of(states: np.ndarray, blocks: np.ndarray) -> list[dict]:
    """Сегменты с постоянной меткой > 0: start, end (exclusive), cls, block."""
    d = np.flatnonzero(np.diff(np.r_[-1, states, -1]) != 0)
    out = []
    for a, b in zip(d[:-1], d[1:]):
        c = int(states[a])
        if c > 0:
            out.append({"start": int(a), "end": int(b), "cls": c, "block": int(blocks[a])})
    return out


def session_info(data_dir: Path = DATA, refresh: bool = False) -> dict[str, dict]:
    """Кэш: для каждой сессии — субъект, длина, NIRS, сегменты (results/cache/sessions.json)."""
    p = CACHE / "sessions.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    import io_utils as iu
    info = {}
    for f in sorted(Path(data_dir).glob("*.mat")):
        s = iu.load_session(f)
        info[f.stem] = {"subject": s["subject"], "n": int(len(s["states"])), "has_nirs": bool(s["has_nirs"]),
                        "fs_nirs": s["fs_nirs"], "segments": segments_of(s["states"], s["blocks"])}
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info))
    return info


def window_context(ends: np.ndarray, segs: list[dict], win: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Для концов окон → (блок, время конца окна от начала сегмента, с)."""
    st = np.array([s["start"] for s in segs])
    i = np.searchsorted(st, np.asarray(ends) - win, side="right") - 1
    blk = np.array([segs[j]["block"] for j in i])
    tin = (np.asarray(ends) - st[i]) / FS
    return blk, tin


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    g = np.zeros((3, 3), dtype=int)
    m = (p >= 1) & (p <= 3) & (y >= 1) & (y <= 3)
    np.add.at(g, (p[m].astype(int) - 1, y[m].astype(int) - 1), 1)
    return g


def f1s(g: np.ndarray) -> np.ndarray:
    tp = np.diag(g).astype(float)
    den = g.sum(1) + g.sum(0)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def min_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1s(confusion(y, p)).min())


def macro_recall(y: np.ndarray, p: np.ndarray) -> float:
    g = confusion(y, p)
    col = g.sum(0)
    return float(np.mean(np.where(col > 0, np.diag(g) / np.maximum(col, 1), 0.0)))


def score_from(F: float, P: float, NF: float | None, pen: float = 0.0) -> float:
    return 60 * F + 30 * P + ((5 + 15 * NF) if NF is not None and not np.isnan(NF) else 0.0) - pen


def load_run(run_dir: Path) -> dict[str, dict]:
    """per_session/*.json прогона."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(run_dir, "per_session").glob("*.json"))}
