"""B1-fix: исходный MVP (legacy/mvp_classifier.py) с двумя исправлениями, включаемыми флагами.

  ema_mode="calls"   — EMA по числу вызовов: a = exp(-1/(tau_s·4)), без Δt (исправление утечки тайминга);
  nirs_scale="fit"   — масштаб NIRS-признаков фиксируется в fit_block и переиспользуется в predict.
При ema_mode="dt", nirs_scale="window" поведение совпадает с MVP. Используется только в экспериментах.

Исходное описание MVP: гибрид ЭЭГ (filter-bank log-power + sLDA) + NIRS (гемодинамические признаки + sLDA).

Вызов: __init__ -> load? -> (push | predict | fit_block)*
Классы: 1=покой, 2=левая, 3=правая. Метки только в fit_block.

Причинность:
  * ЭЭГ фильтруется IIR-фильтрами (scipy.signal.sosfilt) с сохранением состояния zi между push —
    результат идентичен однопроходной причинной фильтрации, будущее не используется.
  * Признаки окна считаются только по отсчётам, уже поступившим через push.
  * Модели обучаются только в fit_block на метках завершённых блоков.
  * Сглаживание — экспоненциальное по времени (в отсчётах ЭЭГ), без знания границ проб.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DEFAULT_CFG: dict[str, Any] = {
    "eeg_bands": [[8, 12], [12, 16], [16, 24], [24, 30]],
    "eeg_drop": ["rpa", "lpa", "fp1", "fp2"],  # опорные/лобные (моргания)
    "win": 250,               # 1 с ЭЭГ
    "train_step": 62,         # шаг нарезки обучающих окон (отсчёты ЭЭГ)
    "train_skip": 0,          # сколько отсчётов пропускать в начале сегмента при обучении
    "nirs_short": 2.0,        # с, «текущий» уровень NIRS
    "nirs_long": 20.0,        # с, скользящий baseline NIRS
    "nirs_lag": 4.0,          # с, дополнительный признак: уровень lag секунд назад
    "w_eeg": 0.8,             # вес ЭЭГ в гибриде (log-вероятности)
    "tau_s": 2.0,             # с, постоянная EMA-сглаживания вероятностей
    "priors": [1 / 3, 1 / 3, 1 / 3],
    "bias": [0.0, 0.0, 0.0],  # добавка к log-вероятностям (калибровка под min-F1)
    "ema_mode": "calls",      # calls (исправлено) | dt (как MVP)
    "nirs_scale": "fit",      # fit (исправлено) | window (как MVP)
}

CLASSES = np.array([1, 2, 3])


class OnlineClassifier:
    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.cfg = dict(DEFAULT_CFG)
        env = os.environ.get("OC_CFG")  # для экспериментов; на проверке не задаётся
        if env:
            self.cfg.update(json.loads(env))
        c = self.cfg
        self.fs = float(meta["fs_eeg"])
        names = list(meta["eeg_channel_names"])
        self.ch = [i for i, n in enumerate(names) if n not in c["eeg_drop"]]
        nmax = int(meta.get("n_samples_eeg", 200_000)) + 1
        self.nb = len(c["eeg_bands"])
        self.sos = [butter(4, b, btype="bandpass", fs=self.fs, output="sos") for b in c["eeg_bands"]]
        self.zi = None
        # кумулятивные суммы квадратов фильтрованного сигнала: S2[t] = sum_{s<t} x_s^2
        self.S2 = np.zeros((nmax, self.nb, len(self.ch)))
        self.t = 0  # сколько отсчётов ЭЭГ получено

        # NIRS
        self.has_nirs = bool(meta.get("has_nirs"))
        self.fs_nirs = float(meta["fs_nirs"]) if self.has_nirs and meta.get("fs_nirs") else None
        self.n_nirs_ch = int(meta.get("n_nirs_ch") or 0)
        self.nirs_t: list[np.ndarray] = []   # время отсчёта в шкале ЭЭГ
        self.nirs_v: list[np.ndarray] = []   # [HbO | HbR]
        self._nt = np.zeros(0)
        self._nC = np.zeros((1, 2 * self.n_nirs_ch))  # кумулятивные суммы
        self._ndirty = False

        self.m_eeg = None
        self.m_nirs = None
        self.train_idx: list[int] = []     # концы обучающих окон
        self.train_y: list[int] = []
        self.p_eeg = None
        self.p_nirs = None
        self.last_t = None
        self.rng = np.random.default_rng(0)
        self._nsc = None

    # ------------------------------------------------------------------ io
    def load(self, artifacts_dir: str | Path) -> None:
        # прогрев (чтобы первый predict не платил за ленивую инициализацию)
        X = self.rng.normal(size=(30, 4))
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, np.repeat([1, 2, 3], 10)).predict_proba(X)

    def push(self, eeg_chunk, nirs_chunk) -> None:
        x = np.asarray(eeg_chunk, dtype=np.float64)
        n = x.shape[0]
        if n:
            x = x[:, self.ch]
            if self.zi is None:
                self.zi = [sosfilt_zi(s)[:, :, None] * x[0][None, None, :] for s in self.sos]
            p = np.empty((n, self.nb, len(self.ch)))
            for b, s in enumerate(self.sos):
                y, self.zi[b] = sosfilt(s, x, axis=0, zi=self.zi[b])
                p[:, b, :] = y * y
            self.S2[self.t + 1 : self.t + n + 1] = self.S2[self.t] + np.cumsum(p, axis=0)
        t0, self.t = self.t, self.t + n

        if self.has_nirs and nirs_chunk is not None:
            hbo, hbr = nirs_chunk
            if hbo is not None and len(hbo):
                k = len(hbo)
                # отсчёты пришли за [t0, t); раскладываем их с номинальным шагом, заканчивая на t
                dt = self.fs / self.fs_nirs
                tt = self.t - dt * np.arange(k - 1, -1, -1)
                tt = np.maximum(tt, t0)
                self.nirs_t.append(tt)
                self.nirs_v.append(np.hstack([hbo, hbr]))
                self._ndirty = True

    # ------------------------------------------------------------ features
    def _eeg_feat(self, ends: np.ndarray) -> np.ndarray:
        w = self.cfg["win"]
        v = (self.S2[ends] - self.S2[ends - w]) / w
        return np.log(v.reshape(len(ends), -1) + 1e-12)

    def _nirs_sync(self) -> None:
        if self._ndirty:
            self._nt = np.concatenate(self.nirs_t)
            V = np.concatenate(self.nirs_v)
            self._nC = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0)])
            self._ndirty = False

    def _nmean(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Среднее NIRS по отсчётам с временем в (a, b]; a,b — векторы (шкала ЭЭГ)."""
        ia = np.searchsorted(self._nt, a, side="right")
        ib = np.searchsorted(self._nt, b, side="right")
        cnt = np.maximum(ib - ia, 1)[:, None]
        return (self._nC[ib] - self._nC[ia]) / cnt

    def _nirs_feat(self, ends: np.ndarray, fit: bool = False) -> np.ndarray:
        self._nirs_sync()
        c, fs = self.cfg, self.fs
        e = ends.astype(float)
        cur = self._nmean(e - c["nirs_short"] * fs, e)
        base = self._nmean(e - c["nirs_long"] * fs, e)
        lag = self._nmean(e - (c["nirs_lag"] + c["nirs_short"]) * fs, e - c["nirs_lag"] * fs)
        F = np.hstack([cur - base, cur - lag])
        if self.cfg["nirs_scale"] == "fit":
            if fit or self._nsc is None:
                sc = np.abs(F).mean() + 1e-12
                self._nsc = sc if np.isfinite(sc) else 1.0
            return F / self._nsc
        sc = np.abs(F).mean() + 1e-12  # масштаб ~1e-5..1e-6 -> единицы (LDA инвариантна, но устойчивее)
        return F / sc if np.isfinite(sc) else F

    # ------------------------------------------------------------- model
    def _fit(self, X: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) < 3:
            return None
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=self.cfg["priors"])
        return m.fit(X, y)

    def fit_block(self, block_idx: int, labels: dict) -> None:
        st = np.asarray(labels["states"])
        idx = np.asarray(labels["eeg_indices"])
        if len(idx) == 0:
            return
        w, step, skip = self.cfg["win"], self.cfg["train_step"], self.cfg["train_skip"]
        # однородные сегменты по индексам (разрывы — метка 0) и смене класса
        brk = np.flatnonzero((np.diff(idx) != 1) | (np.diff(st) != 0)) + 1
        for seg_i, seg_s in zip(np.split(idx, brk), np.split(st, brk)):
            a, b = int(seg_i[0]) + skip, int(seg_i[-1]) + 1
            for end in range(a + w, b + 1, step):
                self.train_idx.append(end)
                self.train_y.append(int(seg_s[0]))
        ends = np.asarray(self.train_idx)
        y = np.asarray(self.train_y)
        self.m_eeg = self._fit(self._eeg_feat(ends), y)
        if self.has_nirs and self.nirs_t:
            self.m_nirs = self._fit(self._nirs_feat(ends, fit=True), y)

    def _proba(self, m, X):
        if m is None:
            return np.full(3, 1 / 3)
        p = np.zeros(3)
        p[np.searchsorted(CLASSES, m.classes_)] = m.predict_proba(X)[0]
        return p

    def predict(self) -> dict:
        end = np.array([self.t])
        pe = self._proba(self.m_eeg, self._eeg_feat(end)) if self.t >= self.cfg["win"] else np.full(3, 1 / 3)
        pn = None
        if self.m_nirs is not None:
            pn = self._proba(self.m_nirs, self._nirs_feat(end))

        # EMA-сглаживание по времени
        a = 0.0
        if self.cfg["ema_mode"] == "calls":
            if self.cfg["tau_s"] > 0:
                a = float(np.exp(-1.0 / (self.cfg["tau_s"] * 4.0)))
        elif self.last_t is not None and self.cfg["tau_s"] > 0:
            a = float(np.exp(-(self.t - self.last_t) / (self.cfg["tau_s"] * self.fs)))
        self.last_t = self.t
        self.p_eeg = pe if self.p_eeg is None else a * self.p_eeg + (1 - a) * pe
        if pn is not None:
            self.p_nirs = pn if self.p_nirs is None else a * self.p_nirs + (1 - a) * pn

        bias = np.asarray(self.cfg["bias"])
        le = np.log(self.p_eeg + 1e-9)
        y_eeg = int(CLASSES[np.argmax(le + bias)])
        out = {"y": y_eeg, "y_eeg": y_eeg, "y_nirs": None}
        if self.has_nirs:
            if self.p_nirs is not None:
                ln = np.log(self.p_nirs + 1e-9)
                w = self.cfg["w_eeg"]
                out["y"] = int(CLASSES[np.argmax(w * le + (1 - w) * ln + bias)])
                out["y_nirs"] = int(CLASSES[np.argmax(ln + bias)])
            else:
                out["y_nirs"] = int(self.rng.integers(1, 4))
        return out
