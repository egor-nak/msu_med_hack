"""Конвертация обученной mne.decoding.CSP (csp_model.joblib коллеги) в numpy-артефакт рантайма.

  python tools/convert_csp.py research/csp/csp_model.joblib solution/artifacts/csp_filters.joblib

Рантайм (solution/hbci/csp.py) не зависит от mne: берём только фильтры filters_[:n_components].
Для mne-CSP с transform_into='csp_space' и log=None: transform(X) = filters_[:n] @ X, а
векторизатор считает log var по времени — это log(w_jᵀ Σ w_j) по ковариации окна Σ.
Порядок каналов в объекте не хранится (info=None): берётся chan_dict из кода коллеги
(C3, Cz, C4, F3, Fz, F4, P3, Pz, P4, C7, C8). Предобработка входа при обучении восстановлена по
patterns_ᵀ·patterns_ (ковариация, в которой нормированы фильтры): дисперсии каналов ~1e14–1e15
отсчётов АЦП² и корреляции 0.98 между каналами → сырые отсчёты без фильтрации ("raw").
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

CHAN_DICT = ["c3", "cz", "c4", "f3", "fz", "f4", "p3", "pz", "p4", "c7", "c8"]


def main(src: str, dst: str, band: str | None = None) -> None:
    """band — "lo,hi" для CSP, обученной на полосовом сигнале (иначе вход "raw")."""
    import mne  # только офлайн (requirements-dev.txt)
    csp = joblib.load(src)
    if not isinstance(csp, mne.decoding.CSP):
        raise TypeError(f"ожидается mne.decoding.CSP, получено {type(csp)}")
    if csp.transform_into != "csp_space" or csp.log is not None:
        raise ValueError("ожидается transform_into='csp_space', log=None (как в векторизаторе коллеги)")
    n = int(csp.n_components)
    if csp.info is not None:                      # имена каналов сохранены в объекте — берём их
        channels = [c.lower() for c in csp.info["ch_names"]]
        ch_source = "mne.Info в объекте"
    else:
        channels = CHAN_DICT
        ch_source = "chan_dict из кода коллеги (в объекте info=None)"
    if csp.filters_.shape[1] != len(channels):
        raise ValueError(f"CSP обучена на {csp.filters_.shape[1]} каналах, список каналов — {len(channels)}")
    Ct = csp.patterns_.T @ csp.patterns_
    # общий средний референс при обучении ⇔ вектор из единиц в ядре обучающей ковариации и Σ w = 0
    one = np.ones(len(channels)) / np.sqrt(len(channels))
    car = bool(np.linalg.norm(Ct @ one) / np.linalg.norm(Ct) < 1e-6
               and (np.abs(csp.filters_.sum(1)) / np.abs(csp.filters_).sum(1)).max() < 1e-4)
    prep = "raw" if band is None else [float(x) for x in band.split(",")]
    art = {
        "filters": np.asarray(csp.filters_[:n], dtype=np.float64),
        "channels": channels,
        "prep": prep,
        "reference": "car" if car else "none",
        "meta": {
            "source": str(src), "type": "mne.decoding.CSP", "mne_version": mne.__version__,
            "params": {k: v for k, v in csp.get_params().items() if k != "info"},
            "classes": np.asarray(csp.classes_).tolist(), "n_features_in": int(csp.n_features_in_),
            "rank": csp.rank if isinstance(csp.rank, (dict, type(None))) else str(csp.rank),
            "sorter": csp.sorter_.tolist(), "mean_power_train": csp.mean_.tolist(), "std_power_train": csp.std_.tolist(),
            "patterns": csp.patterns_[:n].tolist(),
            "train_channel_log10_var": np.log10(np.clip(np.diag(Ct), 1e-300, None)).round(3).tolist(),
            "channel_order_source": ch_source,
            "reference_inferred": "car: ядро patternsᵀ·patterns = вектор из единиц" if car else "нет",
            "train_files": "неизвестно (в объекте не хранится); см. report/CSP_AUDIT.md — отпечаток mean_ совпадает "
                           "со средним по всем 56 сессиям train → оценка на train in-sample",
        },
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, dst)
    print(f"→ {dst}: filters {art['filters'].shape}, channels {channels}, prep={prep}, reference={art['reference']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 2:
        main(a[0], a[1], a[2] if len(a) > 2 else None)
    else:
        main("research/csp/csp_model.joblib", "research/csp/csp_filters_colleague.joblib")
