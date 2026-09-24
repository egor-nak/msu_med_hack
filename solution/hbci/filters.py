"""Потоковые (причинные) фильтры с сохранённым состоянием.

``StreamingSOS`` — обёртка над ``scipy.signal.sosfilt`` с ``zi``: результат фильтрации сигнала,
поданного порциями любого размера, побитово совпадает с однопроходной причинной фильтрацией.
Начальное состояние — установившееся для ступеньки уровня первого отсчёта (``sosfilt_zi · x0``),
это убирает переходный процесс от постоянной составляющей.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt, sosfilt_zi


class StreamingSOS:
    """Причинный SOS-фильтр по оси 0 для многоканального сигнала (n, ...)."""

    def __init__(self, sos: np.ndarray) -> None:
        self.sos = np.asarray(sos, dtype=np.float64)
        self.zi: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return x
        if self.zi is None:
            z0 = sosfilt_zi(self.sos)  # (n_sections, 2)
            self.zi = z0.reshape(z0.shape + (1,) * (x.ndim - 1)) * x[0][None, None, ...]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def butter_sos(order: int, band: float | list[float], fs: float, btype: str) -> np.ndarray:
    return butter(order, band, btype=btype, fs=fs, output="sos")


class FilterBank:
    """Набор полосовых фильтров над одним входом: (n, C) → (n, B, C)."""

    def __init__(self, bands: list[list[float]], fs: float, order: int = 4) -> None:
        self.filters = [StreamingSOS(butter_sos(order, list(b), fs, "bandpass")) for b in bands]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.empty((x.shape[0], len(self.filters), x.shape[1]))
        for i, f in enumerate(self.filters):
            out[:, i, :] = f(x)
        return out


class StreamingEMA:
    """Экспоненциальное среднее по отсчётам: y[n] = a·y[n-1] + (1-a)·x[n] (причинно, с состоянием)."""

    def __init__(self, a: float) -> None:
        self.b = np.array([1.0 - a])
        self.a = np.array([1.0, -a])
        self.zi: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return x
        if self.zi is None:
            self.zi = lfilter_zi(self.b, self.a)[:, None] * x[0][None, :]
        y, self.zi = lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y
