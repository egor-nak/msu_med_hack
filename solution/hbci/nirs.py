"""Фронтенд NIRS: причинные фильтры, базлайн, гемодинамические признаки в шкале отсчётов NIRS.

Сигналы: [HbO(18) | HbR(18) | g_HbO | g_HbR], где g — среднее по каналам (глобальный сигнал).

Синхронизация без зависимости от размеров порций push. Каркас отдаёт вместе с порцией ЭЭГ [t0, t)
все отсчёты NIRS с Frame ≤ t, поэтому число доставленных отсчётов N(t) в моменты вызова predict и
fit_block — свойство данных, а не разбиения на порции. Признаки строятся по «последним L отсчётам
до номера n» (индексная шкала, шаг 1/fs_nirs): в predict n = N(t) — текущее число отсчётов;
для обучающего окна с концом e — N(e), записанное при вызове predict в момент e (все оцениваемые
окна блоков ≥ 2), либо, если вызова не было (блок 1), оценка n = N(T) − round((T − e)/Δ) от
ближайшего записанного момента T ≥ e (Δ = fs_eeg/fs_nirs). Так признаки причинны (используются только
доставленные отсчёты) и инвариантны к разбиению push (tests/test_chunk_invariance.py).

Обработка: ВЧ 0.01 Гц + НЧ 0.4 Гц (Баттерворт 2-го порядка, SOS с состоянием, частота —
meta["fs_nirs"]); EMA-базлайн по отсчётам (τ = base_tau_s).
Признаки для лагов L ∈ lags_s (в отсчётах: L·fs_nirs):
  * уровень: среднее последних level_s·fs_nirs отсчётов минус EMA-базлайн;
  * наклон: МНК-наклон по последним slope_s·fs_nirs отсчётам.
Все признаки линейны по сигналу, поэтому CBSI (x = ½(HbO − α·HbR)), регрессия глобального сигнала
(x − β·g), усреднение по областям и латеральные контрасты — линейная комбинация уже посчитанных
признаков; α и β оцениваются по прошлым данным в fit_block. Знак признаков не фиксируется.
"""

from __future__ import annotations

import numpy as np

from .buffers import GrowingArray
from .filters import StreamingSOS, butter_sos

RIGHT = list(range(0, 8))     # x > 0 по Montage.NIRS_positions
LEFT = list(range(8, 16))     # x < 0
MID = [16, 17]


class NIRSFrontEnd:
    def __init__(self, cfg: dict, fs_eeg: float, fs_nirs: float, n_ch: int) -> None:
        self.cfg = cfg
        self.fs = fs_eeg
        self.fsn = fs_nirs
        self.dt = fs_eeg / fs_nirs
        self.n = n_ch
        self.D = 2 * n_ch + 2
        self.filters: list[StreamingSOS] = []
        if cfg["hp_hz"] > 0:
            self.filters.append(StreamingSOS(butter_sos(2, cfg["hp_hz"], fs_nirs, "highpass")))
        if 0 < cfg["lp_hz"] < fs_nirs / 2:
            self.filters.append(StreamingSOS(butter_sos(2, cfg["lp_hz"], fs_nirs, "lowpass")))
        self.a_base = float(np.exp(-1.0 / (cfg["base_tau_s"] * fs_nirs)))
        self.ema: np.ndarray | None = None
        self.X = GrowingArray((self.D,), 4096)    # отфильтрованный сигнал
        self.E = GrowingArray((self.D,), 4096)    # EMA-базлайн
        self.cx = GrowingArray((self.D,), 4096)   # накопленные суммы x (с ведущим нулём)
        self.cix = GrowingArray((self.D,), 4096)  # накопленные суммы i·x
        self.cx.extend(np.zeros((1, self.D)))
        self.cix.extend(np.zeros((1, self.D)))
        self.rec_t: list[int] = []                # моменты predict/fit_block и N в эти моменты
        self.rec_n: list[int] = []
        self.W: np.ndarray = self._base_W()

    # ------------------------------------------------------------------ push
    def push(self, hbo: np.ndarray, hbr: np.ndarray) -> None:
        k = len(hbo)
        if k == 0:
            return
        hbo = np.asarray(hbo, dtype=np.float64)
        hbr = np.asarray(hbr, dtype=np.float64)
        x = np.hstack([hbo, hbr, hbo.mean(1, keepdims=True), hbr.mean(1, keepdims=True)])
        for f in self.filters:
            x = f(x)
        e = np.empty_like(x)
        s = x[0].copy() if self.ema is None else self.ema
        a = self.a_base
        for i in range(k):
            s = a * s + (1.0 - a) * x[i]
            e[i] = s
        self.ema = s
        i0 = self.X.n
        self.X.extend(x)
        self.E.extend(e)
        idx = np.arange(i0, i0 + k, dtype=np.float64)[:, None]
        self.cx.extend(self.cx.view[-1] + np.cumsum(x, axis=0))
        self.cix.extend(self.cix.view[-1] + np.cumsum(idx * x, axis=0))

    @property
    def n_samples(self) -> int:
        return self.X.n

    def mark(self, t: int) -> None:
        """Запомнить N(t) в момент вызова predict/fit_block (инвариантно к разбиению push)."""
        if self.rec_t and self.rec_t[-1] == t:
            self.rec_n[-1] = self.X.n
            return
        self.rec_t.append(int(t))
        self.rec_n.append(self.X.n)

    def counts_at(self, ends: np.ndarray) -> np.ndarray:
        """Число доставленных отсчётов на момент e (записанное или оценённое от ближайшего T ≥ e)."""
        rt = np.asarray(self.rec_t)
        rn = np.asarray(self.rec_n)
        j = np.searchsorted(rt, ends, side="left")
        j = np.minimum(j, len(rt) - 1)
        T, N = rt[j], rn[j]
        n = N - np.round((T - ends) / self.dt).astype(np.int64)
        return np.clip(n, 0, N)

    # ------------------------------------------------------------- features
    def _mean(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Среднее отсчётов с номерами [a, b)."""
        a = np.clip(a, 0, self.X.n)
        b = np.clip(b, 0, self.X.n)
        cnt = np.maximum(b - a, 1)[:, None]
        return (self.cx.view[b] - self.cx.view[a]) / cnt

    def _slope(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """МНК-наклон (в единицах сигнала за секунду) по отсчётам [a, b)."""
        a = np.clip(a, 0, self.X.n)
        b = np.clip(b, 0, self.X.n)
        n = (b - a).astype(float)[:, None]
        sx = self.cx.view[b] - self.cx.view[a]
        six = self.cix.view[b] - self.cix.view[a]
        si = ((b - 1) * b / 2.0 - (a - 1) * a / 2.0)[:, None]
        si2 = (((b - 1) * b * (2 * b - 1) - (a - 1) * a * (2 * a - 1)) / 6.0)[:, None]
        den = n * si2 - si * si
        ok = (n >= 3) & (den > 1e-9)
        return np.where(ok, (n * six - si * sx) / np.where(ok, den, 1.0), 0.0) * self.fsn

    def _base_at(self, n: np.ndarray) -> np.ndarray:
        i = np.clip(n - 1, 0, max(self.X.n - 1, 0))
        out = self.E.view[i].copy()
        out[n <= 0] = 0.0
        return out

    def raw_features(self, ends: np.ndarray) -> np.ndarray:
        """(N, n_types·n_lags, D) — признаки по исходным сигналам до линейной комбинации."""
        c, fsn = self.cfg, self.fsn
        n = self.counts_at(np.asarray(ends, dtype=np.int64))
        if c["features"] == "mvp":
            sh, lo, lg = (int(round(c[k] * fsn)) for k in ("mvp_short", "mvp_long", "mvp_lag"))
            cur = self._mean(n - sh, n)
            base = self._mean(n - lo, n)
            lag = self._mean(n - lg - sh, n - lg)
            return np.stack([cur - base, cur - lag], axis=1)
        L, Sl = int(round(c["level_s"] * fsn)), int(round(c["slope_s"] * fsn))
        blocks = []
        for lag_s in c["lags_s"]:
            m = n - int(round(lag_s * fsn))
            if c["use_level"]:
                blocks.append(self._mean(m - L, m) - self._base_at(m))
            if c["use_slope"]:
                blocks.append(self._slope(m - Sl, m))
        return np.stack(blocks, axis=1)

    def features(self, ends: np.ndarray) -> np.ndarray:
        F = self.raw_features(ends)
        return (F @ self.W.T).reshape(len(ends), -1)

    # -------------------------------------------------- linear combination
    def _base_W(self, alpha: np.ndarray | None = None, beta: np.ndarray | None = None) -> np.ndarray:
        """Матрица W (n_out, D): строки — итоговые «сигналы» NIRS для признаков."""
        c, n, D = self.cfg, self.n, self.D
        io, ir, go, gr = np.arange(n), n + np.arange(n), 2 * n, 2 * n + 1
        hbo = np.zeros((n, D)); hbo[np.arange(n), io] = 1.0
        hbr = np.zeros((n, D)); hbr[np.arange(n), ir] = 1.0
        if beta is not None and c["gsr"]:
            hbo[:, go] = -beta[:n]
            hbr[:, gr] = -beta[n:]
        if c["features"] == "mvp":
            return np.vstack([hbo, hbr])
        sig = c["signal"]
        if sig == "cbsi":
            a = np.ones(n) if alpha is None else alpha
            chans = [0.5 * (hbo - a[:, None] * hbr)]
        elif sig == "hbo":
            chans = [hbo]
        else:
            chans = [hbo, hbr]
        rows = []
        regions = [RIGHT, LEFT, MID] if n == 18 else [list(range(n))]
        for ch in chans:
            if c["spatial"] == "regions":
                rows += [ch[r].mean(0) for r in regions]
            else:
                rows += list(ch)
        if c["lateral"] and n == 18:
            for ch in chans:
                rows.append(ch[LEFT].mean(0) - ch[RIGHT].mean(0))
        return np.vstack(rows)

    def refit_linear(self) -> None:
        """α (CBSI) и β (GSR) по прошлым отфильтрованным данным; фиксируются до следующего блока."""
        X = self.X.view
        n = self.n
        if len(X) < 10:
            self.W = self._base_W()
            return
        sd = X.std(0) + 1e-12
        alpha = sd[:n] / sd[n : 2 * n]
        g = X[:, [2 * n] * n + [2 * n + 1] * n]
        xc = X[:, : 2 * n] - X[:, : 2 * n].mean(0)
        gc = g - g.mean(0)
        beta = (xc * gc).sum(0) / ((gc * gc).sum(0) + 1e-30)
        self.W = self._base_W(alpha, beta)
