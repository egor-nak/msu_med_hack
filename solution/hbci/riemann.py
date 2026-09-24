"""Операции над SPD-матрицами на numpy (через ``np.linalg.eigh``), без pyriemann в рантайме.

Все функции работают с батчами: ``C`` формы (..., p, p).
  * оценки ковариации с усадкой: OAS, Ledoit–Wolf (из выборочной ковариации и 4-го момента);
  * sqrtm / invsqrtm / logm / expm для симметричных матриц;
  * средние: лог-евклидово и риманово (итерации Карчера);
  * касательное пространство в точке Cref: upper(logm(Cref^-1/2 C Cref^-1/2)), √2 вне диагонали;
  * риманово расстояние.
"""

from __future__ import annotations

import numpy as np


def _eigfn(C: np.ndarray, fn) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _floor(w: np.ndarray) -> np.ndarray:
    return np.maximum(w, 1e-12 * np.max(w, axis=-1, keepdims=True) + 1e-300)


def logm_spd(C: np.ndarray) -> np.ndarray:
    return _eigfn(C, lambda w: np.log(_floor(w)))


def expm_sym(C: np.ndarray) -> np.ndarray:
    return _eigfn(C, np.exp)


def sqrtm_spd(C: np.ndarray) -> np.ndarray:
    return _eigfn(C, lambda w: np.sqrt(_floor(w)))


def invsqrtm_spd(C: np.ndarray) -> np.ndarray:
    return _eigfn(C, lambda w: 1.0 / np.sqrt(_floor(w)))


def _trace(C: np.ndarray) -> np.ndarray:
    return np.trace(C, axis1=-2, axis2=-1)


def regularize(C: np.ndarray, eps: float) -> np.ndarray:
    """C + eps·trace(C)/p·I."""
    p = C.shape[-1]
    return C + (eps * _trace(C) / p)[..., None, None] * np.eye(p)


def shrink_oas(S: np.ndarray, n: int) -> np.ndarray:
    """Oracle Approximating Shrinkage (формула sklearn.covariance.oas) по выборочной ковариации S."""
    p = S.shape[-1]
    mu = _trace(S) / p
    alpha = np.mean(S ** 2, axis=(-2, -1))
    num = alpha + mu ** 2
    den = (n + 1.0) * (alpha - mu ** 2 / p)
    s = np.where(den > 0, np.minimum(num / np.where(den > 0, den, 1.0), 1.0), 1.0)
    return (1.0 - s)[..., None, None] * S + (s * mu)[..., None, None] * np.eye(p)


def shrink_lw(S: np.ndarray, n: int) -> np.ndarray:
    """Ledoit–Wolf (формула sklearn, assume_centered=True) с гауссовым приближением 4-го момента.

    Точная формула требует (1/n)·Σ||x_k||⁴; для гауссовых данных E||x||⁴ = tr(S)² + 2·||S||_F²,
    что позволяет считать усадку только по S (после любого пространственного преобразования).
    """
    p = S.shape[-1]
    tr = _trace(S)
    mu = tr / p
    delta_ = np.sum(S ** 2, axis=(-2, -1))
    q = tr ** 2 + 2.0 * delta_
    beta = (q - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * tr + p * mu ** 2) / p
    beta = np.minimum(beta, delta)
    s = np.where(delta > 0, beta / np.where(delta > 0, delta, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    return (1.0 - s)[..., None, None] * S + (s * mu)[..., None, None] * np.eye(p)


def estimate_cov(S: np.ndarray, n: int, kind: str, eps: float) -> np.ndarray:
    """Оценка ковариации окна по выборочной S: oas | lw | scm, плюс eps·trace/p·I."""
    if kind == "oas":
        C = shrink_oas(S, n)
    elif kind == "lw":
        C = shrink_lw(S, n)
    elif kind == "scm":
        C = S
    else:
        raise ValueError(kind)
    return regularize(C, eps)


def mean_logeuclid(Cs: np.ndarray, axis: int = 0) -> np.ndarray:
    return expm_sym(np.mean(logm_spd(Cs), axis=axis))


def mean_riemann(Cs: np.ndarray, iters: int = 20, tol: float = 1e-6, init: np.ndarray | None = None) -> np.ndarray:
    """Среднее Карчера по оси 0 для Cs (N, ..., p, p)."""
    M = mean_logeuclid(Cs) if init is None else init
    for _ in range(iters):
        Ms, Mis = sqrtm_spd(M), invsqrtm_spd(M)
        T = np.mean(logm_spd(Mis @ Cs @ Mis), axis=0)
        M = Ms @ expm_sym(T) @ Ms
        if np.linalg.norm(T) < tol:
            break
    return M


def mean_spd(Cs: np.ndarray, kind: str, max_n: int = 400) -> np.ndarray:
    """Среднее по оси 0: logeuclid по всем; riemann — Карчер по равномерной подвыборке ≤ max_n."""
    if kind == "logeuclid":
        return mean_logeuclid(Cs)
    if kind == "riemann":
        init = mean_logeuclid(Cs)
        sub = Cs[:: max(1, len(Cs) // max_n)] if len(Cs) > max_n else Cs
        return mean_riemann(sub, iters=20, tol=1e-6, init=init)
    raise ValueError(kind)


def triu_weights(p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iu = np.triu_indices(p)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return iu[0], iu[1], w


def tangent_space(Cs: np.ndarray, Cref_isqrt: np.ndarray) -> np.ndarray:
    """Касательное пространство: Cs (..., p, p), Cref_isqrt = Cref^-1/2 (p, p) → (..., p(p+1)/2)."""
    L = logm_spd(Cref_isqrt @ Cs @ Cref_isqrt)
    i, j, w = triu_weights(Cs.shape[-1])
    return L[..., i, j] * w


def distance_riemann_sq(Cs: np.ndarray, G_isqrt: np.ndarray) -> np.ndarray:
    """d²(C, G) = Σ log² λ(G^-1/2 C G^-1/2)."""
    w = np.linalg.eigvalsh(G_isqrt @ Cs @ G_isqrt)
    return np.sum(np.log(_floor(w)) ** 2, axis=-1)
