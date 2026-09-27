"""CSP-признаки + лог-мощность каналов + межполушарная асимметрия (векторизатор коллеги), numpy-only.

Для окна x ∈ R^{C×T} (C каналов в фиксированном порядке, T = 250 отсчётов):
  1. CSP: y_j = w_jᵀ x — «виртуальные каналы» (пространственные фильтры, строки W);
     признак — log var(y_j) = log(w_jᵀ Σ w_j), Σ — ковариация окна (с вычитанием среднего);
  2. лог-мощность каналов: log var(x_c) = log Σ_cc;
  3. асимметрия пары (L, R): (P_L − P_R)/(P_L + P_R) = tanh(½·(log P_L − log P_R)), P = var окна.

Все три группы — функции одной ковариации Σ окна, поэтому онлайн они считаются из ковариации,
которую фронтенд ЭЭГ уже хранит в точках сетки (EEGFrontEnd.A), без хранения сигнала:
``features_from_cov`` ≡ ``vectorize_windows`` (проверяется тестом tests/test_csp_features.py).

Рантайм не зависит от mne: обученная mne.decoding.CSP конвертируется офлайн (tools/convert_csp.py)
в словарь numpy-массивов {filters, channels, prep, ...} — artifacts/csp_filters.joblib.
CSP здесь НЕ обучается: фильтры фиксированы (обучены офлайн на других данных), поэтому признаки
причинны и не используют меток сессии.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS_REL = 1e-12  # защита log(0): доля от средней величины в своей группе признаков (CSP и каналы — разный масштаб)


def _safe_log(v: np.ndarray) -> np.ndarray:
    """log с нижней границей EPS_REL·mean(v) по строке (масштабы CSP-выходов и мощностей каналов различны)."""
    floor = EPS_REL * np.abs(v).mean(-1, keepdims=True) + 1e-300
    return np.log(np.maximum(v, floor))

# Порядок каналов, на котором обучена CSP коллеги (chan_dict: C3, Cz, C4, F3, Fz, F4, P3, Pz, P4, C7, C8).
CSP_CHANNELS: list[str] = ["c3", "cz", "c4", "f3", "fz", "f4", "p3", "pz", "p4", "c7", "c8"]

# Пары симметричных электродов (левое, правое полушарие). c7/c8 по координатам — T7/T8.
DEFAULT_SYMMETRY: list[tuple[str, str]] = [("c3", "c4"), ("f3", "f4"), ("p3", "p4"), ("c7", "c8")]


def vectorize_windows(data: np.ndarray, filters: np.ndarray, channels: list[str],
                      symmetry: list[tuple[str, str]] | None = DEFAULT_SYMMETRY,
                      use_csp: bool = True, use_power: bool = True) -> np.ndarray:
    """Эталонная (по сигналу) версия векторизатора коллеги с исправлениями.

    data     — (n_windows, n_channels, T), каналы в порядке ``channels``;
    filters  — (n_components, n_channels) — строки CSP-фильтров (= csp.filters_[:n_components]);
    Отличия от исходного кода: асимметрия считается по той же мощности, что и лог-мощность
    (дисперсия окна, а не mean(x²): на сырых данных с постоянной составляющей mean(x²) ≈ DC²,
    и «асимметрия» описывала бы смещение АЦП, а не ритм); проверка размерностей и имён каналов;
    защита log(0); без np.concatenate в цикле.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or data.shape[1] != len(channels):
        raise ValueError(f"ожидается (n, {len(channels)}, T), получено {data.shape}")
    if filters.shape[1] != data.shape[1]:
        raise ValueError(f"CSP обучена на {filters.shape[1]} каналах, данные — {data.shape[1]}")
    xc = data - data.mean(-1, keepdims=True)
    power = (xc ** 2).mean(-1)                                  # (n, C) = var по окну
    parts = []
    if use_csp:
        y = np.einsum("jc,nct->njt", filters, xc)
        parts.append(_safe_log((y ** 2).mean(-1)))
    if use_power:
        parts.append(_safe_log(power))
    if symmetry:
        idx = {c: i for i, c in enumerate(channels)}
        L = power[:, [idx[a] for a, _ in symmetry]]
        R = power[:, [idx[b] for _, b in symmetry]]
        den = L + R
        parts.append(np.divide(L - R, den, out=np.zeros_like(den), where=den > 0))
    return np.concatenate(parts, axis=1)


def features_from_cov(C: np.ndarray, filters: np.ndarray, channels: list[str],
                      symmetry: list[tuple[str, str]] | None = DEFAULT_SYMMETRY,
                      use_csp: bool = True, use_power: bool = True) -> np.ndarray:
    """То же по ковариации окна C (N, n_ch, n_ch) (с вычитанием среднего окна): точный эквивалент."""
    C = np.asarray(C, dtype=np.float64)
    power = np.diagonal(C, axis1=-2, axis2=-1)
    parts = []
    if use_csp:
        parts.append(_safe_log(np.einsum("jc,ncd,jd->nj", filters, C, filters)))
    if use_power:
        parts.append(_safe_log(power))
    if symmetry:
        idx = {c: i for i, c in enumerate(channels)}
        L = power[:, [idx[a] for a, _ in symmetry]]
        R = power[:, [idx[b] for _, b in symmetry]]
        den = L + R
        parts.append(np.divide(L - R, den, out=np.zeros_like(den), where=den > 0))
    return np.concatenate(parts, axis=1)


def csp_models(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Список CSP-моделей из секции eeg.csp: {"models": [{"file", "prep"}, ...]} или прежний формат {"file", "prep"}."""
    if cfg.get("models"):
        return [dict(m) for m in cfg["models"]]
    return [{"file": cfg.get("file", "csp_filters.joblib"), "prep": cfg.get("prep", "raw")}]


@dataclass
class CSPArtifact:
    """Обученная (офлайн) CSP в виде numpy: фильтры, порядок каналов, требуемая предобработка."""

    filters: np.ndarray          # (n_components, n_ch)
    channels: list[str]          # порядок входных каналов (нижний регистр, имена io_utils)
    prep: Any                    # "raw" | [lo, hi] — вход, на котором CSP обучалась
    meta: dict[str, Any]
    reference: str = "none"      # "none" | "car" — общий средний референс по self.channels перед CSP

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CSPArtifact":
        return cls(np.asarray(d["filters"], dtype=np.float64), [str(c).lower() for c in d["channels"]],
                   d["prep"], dict(d.get("meta", {})), str(d.get("reference", "none")))

    def check(self, all_names: list[str], prep: Any) -> None:
        missing = [c for c in self.channels if c not in all_names]
        if missing:
            raise ValueError(f"CSP: нет каналов {missing} в данных")
        if self.filters.shape[1] != len(self.channels):
            raise ValueError("CSP: число каналов фильтров ≠ списку каналов")
        norm = (lambda p: p if p == "raw" else [float(x) for x in p])
        if norm(self.prep) != norm(prep):
            raise ValueError(f"CSP обучена на входе {self.prep!r}, а фронтенд считает {prep!r}")
