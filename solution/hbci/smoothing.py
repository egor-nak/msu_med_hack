"""Сглаживание по числу вызовов ``predict`` (без Δt).

a = exp(-1/(tau_s·calls_per_s)); tau_s = 0 — сглаживание выключено. Коэффициент не зависит от времени
между вызовами: пауза между сегментами (которая раскрывает границу сегмента) не влияет на ответ.
"""

from __future__ import annotations

import numpy as np


class CallEMA:
    def __init__(self, tau_s: float, calls_per_s: float = 4.0, space: str = "log") -> None:
        self.a = float(np.exp(-1.0 / (tau_s * calls_per_s))) if tau_s > 0 else 0.0
        self.space = space
        self.state: np.ndarray | None = None

    def __call__(self, logp: np.ndarray) -> np.ndarray:
        """Вход — лог-вероятности; выход — сглаженные лог-вероятности (нормированные)."""
        v = np.exp(logp) if self.space == "prob" else logp
        self.state = v.copy() if self.state is None else self.a * self.state + (1.0 - self.a) * v
        if self.space == "prob":
            return np.log(self.state + 1e-9)
        s = self.state - self.state.max()
        return s - np.log(np.exp(s).sum())


def ema_sequence(logp: np.ndarray, a: float, space: str = "log") -> np.ndarray:
    """EMA по последовательности (N, 3) лог-вероятностей (для калибровки на вне-выборочных окнах)."""
    v = np.exp(logp) if space == "prob" else logp
    out = np.empty_like(v)
    s = v[0]
    for i in range(len(v)):
        s = v[i] if i == 0 else a * s + (1.0 - a) * v[i]
        out[i] = s
    if space == "prob":
        return np.log(out + 1e-9)
    out = out - out.max(1, keepdims=True)
    return out - np.log(np.exp(out).sum(1, keepdims=True))
