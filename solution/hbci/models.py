"""Модели: линейные (sLDA / LR / каскад) в виде numpy-коэффициентов, MDM, CSP, глобальная ЭЭГ-модель.

Обучение — через scikit-learn (только в fit_block / train.py); предсказание — чистый numpy
(стандартизация + линейная функция + log-softmax): это быстро и не зависит от накладных
расходов sklearn в predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh as geigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from .riemann import distance_riemann_sq, invsqrtm_spd, mean_spd

CLASSES = np.array([1, 2, 3])


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


@dataclass
class LinearModel:
    """log p = log_softmax(((X - mu)/sd) @ W.T + b); W: (3, d)."""

    mu: np.ndarray
    sd: np.ndarray
    W: np.ndarray
    b: np.ndarray

    def logp(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        return log_softmax(Z @ self.W.T + self.b)


@dataclass
class CascadeModel:
    """Каскад: p(покой) бинарной моделью, затем p(Л|воображение) второй бинарной моделью."""

    mu: np.ndarray
    sd: np.ndarray
    w1: np.ndarray
    b1: float
    w2: np.ndarray
    b2: float

    def logp(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        z1 = Z @ self.w1 + self.b1          # logit p(покой)
        z2 = Z @ self.w2 + self.b2          # logit p(правая | воображение)
        lr = -np.logaddexp(0.0, -z1)        # log p(покой)
        li = -np.logaddexp(0.0, z1)         # log p(воображение)
        l3 = -np.logaddexp(0.0, -z2)
        l2 = -np.logaddexp(0.0, z2)
        return np.stack([lr, li + l2, li + l3], axis=1)


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd, mu, sd


def _norm_logits(m: "LinearModel", X: np.ndarray, T: float) -> "LinearModel":
    """Нормировка масштаба логитов: центрированные логиты на обучении → SD = 1, затем / T.

    sLDA на сотнях признаков при сотнях окон даёт логиты порядка 10²: такие «уверенные» вероятности
    подавляют остальные источники при смешивании/слиянии и делают сетку смещений [−1.5, 1.5] бессмысленной.
    Нормировка сохраняет argmax и ранжирование, меняет только шкалу (аналог temperature scaling).
    """
    Z = ((X - m.mu) / m.sd) @ m.W.T + m.b
    Zc = Z - Z.mean(1, keepdims=True)
    s = float(Zc.std()) * T
    if s > 1e-12:
        m.W = m.W / s
        m.b = m.b / s
    return m


def fit_linear(X: np.ndarray, y: np.ndarray, kind: str = "slda", C: float = 0.1,
               priors: list[float] | None = None, random_state: int = 0,
               norm_T: float | None = None) -> LinearModel | CascadeModel | None:
    """Обучение на стандартизованных признаках; None, если в выборке < 3 классов.

    norm_T — если задано, логиты нормируются (см. _norm_logits) с температурой norm_T."""
    if len(np.unique(y)) < 3 or len(y) < 6:
        return None
    Z, mu, sd = _standardize(X)
    if kind == "cascade":
        yr = (y == 1).astype(int)
        m1 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(Z, yr)
        im = y != 1
        m2 = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(Z[im], (y[im] == 3).astype(int))
        return CascadeModel(mu, sd, m1.coef_[0].copy(), float(m1.intercept_[0]), m2.coef_[0].copy(), float(m2.intercept_[0]))
    if kind == "slda":
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors or [1 / 3] * 3).fit(Z, y)
    elif kind == "lr":
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=300, random_state=random_state).fit(Z, y)
    else:
        raise ValueError(kind)
    W = np.zeros((3, Z.shape[1]))
    b = np.full(3, -30.0)
    pos = np.searchsorted(CLASSES, m.classes_)
    W[pos] = m.coef_
    b[pos] = m.intercept_
    lm = LinearModel(mu, sd, W, b)
    return _norm_logits(lm, X, norm_T) if norm_T else lm


# ------------------------------------------------------------------------ MDM
@dataclass
class MDMModel:
    """Minimum Distance to Mean в пространстве, перецентрированном в Cref (по полосам)."""

    G_isqrt: np.ndarray  # (3, B, p, p) — центроиды^-1/2
    T: float

    def logp(self, Cw: np.ndarray) -> np.ndarray:
        """Cw: (N, B, p, p) — перецентрированные ковариации."""
        d2 = np.stack([distance_riemann_sq(Cw, self.G_isqrt[c]).sum(-1) for c in range(3)], axis=-1)
        return log_softmax(-d2 / self.T)


def fit_mdm(Cw: np.ndarray, y: np.ndarray, mean_kind: str, T_mult: float, max_n: int) -> MDMModel | None:
    if len(np.unique(y)) < 3:
        return None
    G = np.stack([mean_spd(Cw[y == c], mean_kind, max_n) for c in CLASSES])  # (3, B, p, p)
    Gi = invsqrtm_spd(G)
    m = MDMModel(Gi, 1.0)
    d2 = -m.logp(Cw)  # при T=1 это d² минус константа по строке
    d2 = d2 - d2.min(1, keepdims=True)
    scale = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    m.T = max(T_mult * scale, 1e-6)
    return m


@dataclass
class TSMDMModel:
    """MDM с лог-евклидовой метрикой в касательном пространстве (в Cref): ближайшее среднее класса.

    d²(C, G_c) = ||t(C) − mean_c t||² — лог-евклидово расстояние между перецентрированными ковариациями,
    где t — касательный вектор (√2 вне диагонали, т.е. ||t||² = ||logm||_F²). Требует только уже посчитанных
    TS-векторов (без дополнительных разложений), поэтому дёшев и в fit_block, и при LOBO-калибровке.
    """

    G: np.ndarray  # (3, d)
    T: float

    def logp(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        d2 = (X * X).sum(1)[:, None] - 2.0 * X @ self.G.T + (self.G * self.G).sum(1)[None, :]
        return log_softmax(-d2 / self.T)


def fit_ts_mdm(X: np.ndarray, y: np.ndarray, T_mult: float) -> TSMDMModel | None:
    if len(np.unique(y)) < 3:
        return None
    G = np.stack([X[y == c].mean(0) for c in CLASSES])
    m = TSMDMModel(G, 1.0)
    d2 = -m.logp(X)
    d2 = d2 - d2.min(1, keepdims=True)
    scale = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    m.T = max(T_mult * scale, 1e-6)
    return m


# ------------------------------------------------------------------------ CSP
def fit_csp(C: np.ndarray, y: np.ndarray, n_per_class: int) -> np.ndarray | None:
    """Multiclass CSP (one-vs-rest) по полосам. C: (N, B, p, p) → фильтры (B, p, 3·n_per_class)."""
    if len(np.unique(y)) < 3:
        return None
    B, p = C.shape[1], C.shape[2]
    out = np.zeros((B, p, 3 * n_per_class))
    for b in range(B):
        cols = []
        for c in CLASSES:
            Ca = C[y == c, b].mean(0)
            Cb = C[y != c, b].mean(0)
            w, V = geigh(Ca, Ca + Cb)
            hi = n_per_class - n_per_class // 2
            lo = n_per_class // 2
            sel = list(range(p - 1, p - 1 - hi, -1)) + list(range(lo))
            cols.append(V[:, sel])
        out[b] = np.concatenate(cols, axis=1)
    return out


def csp_features(C: np.ndarray, Wf: np.ndarray) -> np.ndarray:
    """log-дисперсия CSP-проекций: C (N, B, p, p), Wf (B, p, m) → (N, B·m)."""
    v = np.einsum("bpm,nbpq,bqm->nbm", Wf, C, Wf)
    return np.log(np.maximum(v, 1e-30)).reshape(C.shape[0], -1)


# ----------------------------------------------------------------- global EEG
@dataclass
class GlobalEEG:
    """Межсубъектная модель (TS + стандартизация + LR), обученная train.py.

    Перецентрирование: Cref = среднее ковариаций окон блока 1 текущей сессии (фиксируется
    при первом fit_block) — так же, как при обучении (Cref_session по блоку 1).
    """

    meta: dict[str, Any]
    model: LinearModel

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GlobalEEG":
        return cls(d["meta"], LinearModel(np.asarray(d["mu"]), np.asarray(d["sd"]),
                                          np.asarray(d["W"]), np.asarray(d["b"])))

    def check_compatible(self, eeg_cfg: dict[str, Any], channels: list[str]) -> None:
        m = self.meta
        for key in ("reference", "cov", "hp_hz", "band_order", "cov_avg_n"):
            if m["eeg"].get(key, 1) != eeg_cfg.get(key, 1):
                raise ValueError(f"глобальная модель несовместима: {key}={m['eeg'][key]} ≠ {eeg_cfg[key]}")
        if [list(map(float, b)) for b in m["eeg"]["bands"]] != [list(map(float, b)) for b in eeg_cfg["bands"]]:
            raise ValueError("глобальная модель несовместима: полосы")
        if m["channels"] != channels:
            raise ValueError("глобальная модель несовместима: каналы")
