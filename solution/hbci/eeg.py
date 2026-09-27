"""Фронтенд ЭЭГ: причинная фильтрация и ковариации окон на фиксированной сетке.

push(x):
  1. вычитается первый отсчёт сессии (константа, известна с первого push) — точность float64;
  2. ВЧ Баттерворт 2-го порядка (``hp_hz``) по всем 21 каналу, SOS с состоянием;
  3. банк полосовых фильтров (Баттерворт ``band_order``) → (n, B, 21);
  4. в каждой точке сетки конца окна e = win + k·step (та же сетка, что у окон run.py) —
     выборочная ковариация окна [e−win, e) по всем 21 каналу для каждой полосы.

Пространственный референс (none | car_clean | laplacian) и выбор каналов — линейное
преобразование M (p × 21), применяемое в пространстве ковариаций: C' = M·S·Mᵀ. Поэтому маску
плохих каналов (по прошлым данным, в fit_block) можно учесть и в референсе: плохой канал не
входит в CAR/лапласиан. Для глобальной модели плохой канал интерполируется соседями
(размерность сохраняется).

Ответы не зависят от размеров порций push: фильтрация с состоянием побитово инвариантна
к разбиению, а ковариации считаются только в точках сетки.
"""

from __future__ import annotations

import sys

import numpy as np

from .config import CAR_EXCLUDE, LAPLACIAN, resolve_channels
from .filters import FilterBank, StreamingEMA, StreamingSOS, butter_sos

# Соседи для интерполяции плохих каналов (c7/c8 ≈ T7/T8).
NEIGHBORS: dict[str, list[str]] = {
    "c3": ["f3", "p3", "c7", "cz"], "c4": ["f4", "p4", "c8", "cz"], "cz": ["fz", "pz", "c3", "c4"],
    "f3": ["fz", "c3", "f7", "fp1"], "f4": ["fz", "c4", "f8", "fp2"], "fz": ["f3", "f4", "cz"],
    "p3": ["pz", "c3", "p7", "o1"], "p4": ["pz", "c4", "p8", "o2"], "pz": ["p3", "p4", "cz"],
    "c7": ["c3", "f7", "p7"], "c8": ["c4", "f8", "p8"], "f7": ["fp1", "f3", "c7"], "f8": ["fp2", "f4", "c8"],
    "p7": ["c7", "p3", "o1"], "p8": ["c8", "p4", "o2"], "o1": ["pz", "p3", "o2"], "o2": ["pz", "p4", "o1"],
    "fp1": ["f3", "fz", "fp2"], "fp2": ["f4", "fz", "fp1"], "lpa": ["c7", "p7", "f7"], "rpa": ["c8", "p8", "f8"],
}


def spatial_matrix(names: list[str], out: list[str], reference: str, bad: set[str] | None = None,
                   interpolate: bool = False) -> tuple[np.ndarray, list[str]]:
    """Матрица M (p, 21) и список итоговых каналов.

    bad — плохие каналы: исключаются из CAR/лапласиана; если interpolate — выходной плохой канал
    заменяется средним хороших соседей, иначе удаляется из выхода.
    """
    bad = bad or set()
    idx = {n: i for i, n in enumerate(names)}
    n_in = len(names)
    raw = np.eye(n_in)
    for c in bad:
        nb = [m for m in NEIGHBORS.get(c, []) if m not in bad]
        if interpolate and nb:
            raw[idx[c]] = 0.0
            raw[idx[c], [idx[m] for m in nb]] = 1.0 / len(nb)
    base = raw.copy()
    if reference == "car_clean":
        clean = [n for n in names if n not in CAR_EXCLUDE and n not in bad]
        car = raw[[idx[n] for n in clean]].mean(0)
        base = raw - car[None, :]
    elif reference == "laplacian":
        for c, nb in LAPLACIAN.items():
            nbg = [m for m in nb if m not in bad]
            if c in bad or not nbg:
                continue
            base[idx[c]] = raw[idx[c]] - raw[[idx[m] for m in nbg]].mean(0)
    elif reference != "none":
        raise ValueError(f"reference={reference!r}")
    keep = [c for c in out if interpolate or c not in bad]
    return np.stack([base[idx[c]] for c in keep]), keep


def transform(S: np.ndarray, M: np.ndarray) -> np.ndarray:
    """C' = M S Mᵀ для батча S (..., 21, 21)."""
    return M @ S @ M.T


class EEGFrontEnd:
    """Хранит ковариации окон (K, B, 21, 21) в точках сетки и статистики для маски каналов."""

    def __init__(self, cfg: dict, fs: float, names: list[str], n_samples: int, win: int, step: int) -> None:
        self.cfg = cfg
        self.fs = fs
        self.names = list(names)
        self.N = len(self.names)
        self.win, self.step = int(win), int(step)
        self.channels = resolve_channels(cfg["channels"])
        self.bands = [list(map(float, b)) for b in cfg["bands"]]
        self.B = len(self.bands)
        self.hp = StreamingSOS(butter_sos(2, cfg["hp_hz"], fs, "highpass")) if cfg["hp_hz"] > 0 else None
        self.bank = FilterBank(self.bands, fs, int(cfg["band_order"]))
        kmax = n_samples // self.step + 4
        self.S = np.zeros((kmax, self.B, self.N, self.N))   # X^T X / win по всем 21 каналу
        self.ends = np.zeros(kmax, dtype=np.int64)
        self.K = 0
        self.t = 0
        self.x0: np.ndarray | None = None
        self.tail = np.zeros((0, self.B, self.N))
        self.flat = np.zeros(self.N)
        self.nflat = 0
        self.last_raw: np.ndarray | None = None
        # вспомогательная ковариация для CSP-признаков (hbci/csp.py): вход CSP должен совпадать с тем,
        # на чём CSP обучалась. "raw" — сырые отсчёты АЦП (x − x0) без фильтров; [lo, hi] — причинный
        # полосовой фильтр Баттерворта 4-го порядка после того же ВЧ. Ковариация окна — с вычитанием
        # среднего окна (как np.var в векторизаторе). None — не считается.
        csp = cfg.get("csp") or {}
        self.aux = csp.get("prep") if csp.get("enabled") else None
        self.aux_f = None
        if self.aux is not None and self.aux != "raw":
            self.aux_f = StreamingSOS(butter_sos(4, list(map(float, self.aux)), fs, "bandpass"))
        self.A = np.zeros((kmax, self.N, self.N)) if self.aux is not None else None
        self.aux_tail = np.zeros((0, self.N))
        # контрольный heog-признак (f7−f8, fp1−fp2; 0.1–3 Гц; минус EMA τ) — только B-conf
        self.heog_on = cfg["features"] == "heog"
        if self.heog_on:
            idx = {n: i for i, n in enumerate(self.names)}
            self.heog_M = np.zeros((2, self.N))
            self.heog_M[0, idx["f7"]], self.heog_M[0, idx["f8"]] = 1, -1
            self.heog_M[1, idx["fp1"]], self.heog_M[1, idx["fp2"]] = 1, -1
            self.heog_f = StreamingSOS(butter_sos(2, cfg["heog_band"], fs, "bandpass"))
            self.heog_ema = StreamingEMA(float(np.exp(-1.0 / (cfg["heog_tau_s"] * fs))))
            self.heog_tail = np.zeros((0, 2))
            self.heog_ema_last = np.zeros(2)
            self.H = np.zeros((kmax, 2))

    # ------------------------------------------------------------------ push
    def push(self, x: np.ndarray) -> None:
        n = x.shape[0]
        if n == 0:
            return
        x = np.asarray(x, dtype=np.float64)
        if self.x0 is None:
            self.x0 = x[0].copy()
        prev = x[:1] if self.last_raw is None else self.last_raw[None, :]
        self.flat += (np.diff(np.vstack([prev, x]), axis=0) == 0).sum(0)
        self.nflat += n
        self.last_raw = x[-1].copy()
        z = x - self.x0
        if self.aux == "raw":
            abuf = np.concatenate([self.aux_tail, z], axis=0)
        if self.hp is not None:
            z = self.hp(z)
        if self.aux_f is not None:
            abuf = np.concatenate([self.aux_tail, self.aux_f(z)], axis=0)
        y = self.bank(z)                                  # (n, B, 21)
        buf = np.concatenate([self.tail, y], axis=0)      # buf[0] ↔ отсчёт t − len(tail)
        base = self.t - len(self.tail)
        t_new = self.t + n
        k0 = max(0, -(-(self.t + 1 - self.win) // self.step))
        ends = self.win + self.step * np.arange(k0, (t_new - self.win) // self.step + 1)
        ends = ends[ends > self.t]
        if self.heog_on:
            h = self.heog_f(z @ self.heog_M.T)
            he = self.heog_ema(h)
            hbuf = np.concatenate([self.heog_tail, h])
        for e in ends:
            a = e - self.win - base
            X = buf[a : a + self.win].transpose(1, 0, 2)  # (B, win, 21)
            k = self.K
            self.S[k] = np.swapaxes(X, 1, 2) @ X / self.win
            if self.A is not None:
                Aw = abuf[a : a + self.win]
                Aw = Aw - Aw.mean(0)
                self.A[k] = Aw.T @ Aw / self.win
            self.ends[k] = e
            if self.heog_on:
                self.H[k] = hbuf[a : a + self.win].mean(0) - he[e - 1 - self.t]
            self.K += 1
        self.tail = buf[-self.win :]
        if self.A is not None:
            self.aux_tail = abuf[-self.win :]
        if self.heog_on:
            self.heog_tail = hbuf[-self.win :]
            self.heog_ema_last = he[-1]
        self.t = t_new

    # --------------------------------------------------------------- access
    def index_of(self, end: int) -> int | None:
        if end < self.win or (end - self.win) % self.step:
            return None
        k = (end - self.win) // self.step
        return k if k < self.K and self.ends[k] == end else None

    def current_aux(self) -> np.ndarray | None:
        """Вспомогательная (CSP) ковариация окна, заканчивающегося на текущем t."""
        if self.A is None:
            return None
        k = self.index_of(self.t)
        if k is not None:
            return self.A[k]
        Aw = self.aux_tail[-self.win :]
        Aw = Aw - Aw.mean(0)
        return Aw.T @ Aw / max(len(Aw), 1)

    def current(self) -> tuple[np.ndarray, np.ndarray | None]:
        """(S, heog) окна, заканчивающегося на текущем t (из сетки или напрямую из хвоста)."""
        k = self.index_of(self.t)
        if k is not None:
            return self.S[k], (self.H[k] if self.heog_on else None)
        X = self.tail[-self.win :].transpose(1, 0, 2)
        S = np.swapaxes(X, 1, 2) @ X / max(X.shape[1], 1)
        H = None
        if self.heog_on:
            H = self.heog_tail[-self.win :].mean(0) - self.heog_ema_last
        return S, H

    # ------------------------------------------------------- bad channels
    def bad_channels(self) -> set[str]:
        """Плохие каналы (из всех 21) по прошлым данным.

        * доля «плоских» отсчётов (diff == 0) > bad_flat_frac — клиппинг/обрыв (из-за дрейфа DC
          критерий |x| > 0.99·max на этих данных неинформативен, клиппинг виден как плато);
        * медианная (по окнам) мощность канала 4–30 Гц отличается от медианы по каналам более чем
          в bad_ratio раз вверх (шумный контакт) или в 50·bad_ratio раз вниз (обрыв).
        """
        if not self.cfg.get("bad_channels", True) or self.K < 8:
            return set()
        flat = self.flat / max(self.nflat, 1)
        v = np.log(np.einsum("kbcc->kc", self.S[: self.K]) + 1e-30)
        med = np.median(v, axis=0)
        d = med - np.median(med)
        r = float(self.cfg["bad_ratio"])
        bad = {self.names[i] for i in range(self.N)
               if flat[i] > self.cfg["bad_flat_frac"] or d[i] > np.log(r) or d[i] < -np.log(50 * r)}
        for c in ("c3", "c4"):
            if c in bad and c in self.channels:
                print(f"[hbci] предупреждение: канал {c} исключён как плохой", file=sys.stderr)
        return bad
