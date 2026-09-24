"""Слияние модальностей и калибровка смещений логитов под min-F1.

Слияние (на сглаженных лог-вероятностях):
  * loglinear: log p = w·log p_eeg + (1−w)·log p_nirs (затем нормировка);
  * avg:       log p = log(½(p_eeg + p_nirs));
  * stacking:  multinomial LR на [log p_eeg, log p_nirs] (6 признаков), веса обучены офлайн
               на вне-выборочных (LOSO) предсказаниях и хранятся в fusion.json.
Без NIRS — чистая ЭЭГ-ветка.

Смещение b = (0, b2, b3) прибавляется к логитам перед argmax; подбирается координатным спуском
по сетке так, чтобы максимизировать min-F1 (среднее по группам-сессиям, если они заданы).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import log_softmax


def fuse(le: np.ndarray, ln: np.ndarray | None, mode: str, w_eeg: float = 0.8,
         stack: dict[str, Any] | None = None) -> np.ndarray:
    """le, ln: (..., 3) лог-вероятности → (..., 3)."""
    if ln is None:
        return le
    if mode == "loglinear":
        return log_softmax(w_eeg * le + (1.0 - w_eeg) * ln)
    if mode == "avg":
        return np.log(0.5 * (np.exp(le) + np.exp(ln)) + 1e-12)
    if mode == "stacking":
        if stack is None:
            return log_softmax(w_eeg * le + (1.0 - w_eeg) * ln)
        Z = np.concatenate([le, ln], axis=-1)
        Z = (Z - np.asarray(stack["mu"])) / np.asarray(stack["sd"])
        return log_softmax(Z @ np.asarray(stack["W"]).T + np.asarray(stack["b"]))
    raise ValueError(mode)


def min_f1_many(pred: np.ndarray, y: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """pred: (M, N) классы 0..2 для M кандидатов; y: (N,) 0..2 → (M,) min-F1 (среднее по группам)."""
    M, N = pred.shape
    if groups is None:
        groups = np.zeros(N, dtype=int)
    _, g = np.unique(groups, return_inverse=True)
    G = g.max() + 1
    # confusion[m, grp, p, t]
    code = ((np.arange(M)[:, None] * G + g[None, :]) * 3 + pred) * 3 + y[None, :]
    cm = np.bincount(code.ravel(), minlength=M * G * 9).reshape(M, G, 3, 3)
    tp = np.diagonal(cm, axis1=2, axis2=3)
    den = cm.sum(3) + cm.sum(2)
    f1 = np.where(den > 0, 2.0 * tp / np.maximum(den, 1), 0.0)
    return f1.min(2).mean(1)


def calibrate_bias(logits: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                   grid: tuple[float, float, float] = (-1.5, 1.5, 0.1), passes: int = 2,
                   init: np.ndarray | None = None) -> np.ndarray:
    """Координатный спуск по b2, b3 (b1 = 0), максимизирующий min-F1. y — метки 1..3."""
    lo, hi, st = grid
    vals = np.round(np.arange(lo, hi + st / 2, st), 6)
    b = np.zeros(3) if init is None else np.asarray(init, dtype=float).copy()
    b[0] = 0.0
    yy = np.asarray(y) - 1
    for _ in range(passes):
        for j in (1, 2):
            cand = np.repeat(b[None, :], len(vals), axis=0)
            cand[:, j] = vals
            pred = np.argmax(logits[None, :, :] + cand[:, None, :], axis=2)
            sc = min_f1_many(pred, yy, groups)
            best = np.flatnonzero(sc >= sc.max() - 1e-12)
            # при равенстве — ближайшее к нулю (устойчивость)
            b[j] = vals[best[np.argmin(np.abs(vals[best]))]]
    return b
