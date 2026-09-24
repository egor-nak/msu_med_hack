"""Буферы: растущий массив с амортизированным добавлением и NIRS-таймстемпы."""

from __future__ import annotations

import numpy as np


class GrowingArray:
    """Массив (n, *shape), растущий по оси 0; ``view`` — заполненная часть без копии."""

    def __init__(self, shape: tuple[int, ...], capacity: int = 1024) -> None:
        self.data = np.zeros((max(capacity, 1),) + tuple(shape))
        self.n = 0

    def extend(self, x: np.ndarray) -> None:
        k = x.shape[0]
        if self.n + k > self.data.shape[0]:
            cap = max(self.data.shape[0] * 2, self.n + k)
            new = np.zeros((cap,) + self.data.shape[1:])
            new[: self.n] = self.data[: self.n]
            self.data = new
        self.data[self.n : self.n + k] = x
        self.n += k

    @property
    def view(self) -> np.ndarray:
        return self.data[: self.n]


def nirs_timestamps(k: int, t0: int, t1: int, fs_eeg: float, fs_nirs: float) -> np.ndarray:
    """Время k отсчётов NIRS, пришедших в порции ЭЭГ [t0, t1), в шкале отсчётов ЭЭГ.

    Раскладываем с номинальным шагом fs_eeg/fs_nirs, заканчивая на t1 (не позже конца порции:
    отсчёт не может «появиться» раньше, чем доставлен) и не раньше t0.
    """
    dt = fs_eeg / fs_nirs
    tt = t1 - dt * np.arange(k - 1, -1, -1, dtype=np.float64)
    return np.maximum(tt, float(t0))
