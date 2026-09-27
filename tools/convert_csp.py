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


def main(src: str, dst: str) -> None:
    import mne  # только офлайн (requirements-dev.txt)
    csp = joblib.load(src)
    if not isinstance(csp, mne.decoding.CSP):
        raise TypeError(f"ожидается mne.decoding.CSP, получено {type(csp)}")
    n = int(csp.n_components)
    if csp.filters_.shape[1] != len(CHAN_DICT):
        raise ValueError(f"CSP обучена на {csp.filters_.shape[1]} каналах, chan_dict — {len(CHAN_DICT)}")
    if csp.transform_into != "csp_space" or csp.log is not None:
        raise ValueError("ожидается transform_into='csp_space', log=None (как в векторизаторе коллеги)")
    Ct = csp.patterns_.T @ csp.patterns_
    art = {
        "filters": np.asarray(csp.filters_[:n], dtype=np.float64),
        "channels": CHAN_DICT,
        "prep": "raw",
        "meta": {
            "source": str(src), "type": "mne.decoding.CSP", "mne_version": mne.__version__,
            "params": csp.get_params(), "classes": csp.classes_.tolist(), "n_features_in": int(csp.n_features_in_),
            "sorter": csp.sorter_.tolist(), "mean_power_train": csp.mean_.tolist(), "std_power_train": csp.std_.tolist(),
            "patterns": csp.patterns_[:n].tolist(),
            "train_channel_log10_var": np.log10(np.diag(Ct)).round(3).tolist(),
            "channel_order_source": "chan_dict из кода коллеги (в объекте info=None)",
            "prep_inferred_from": "patterns_.T @ patterns_: var ~1e14-1e15 ADC^2, corr 0.98 → сырые отсчёты",
            "train_files": "неизвестно (в объекте не хранится) — для оценки на train возможна утечка in-sample",
        },
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, dst)
    print(f"→ {dst}: filters {art['filters'].shape}, channels {CHAN_DICT}, prep=raw")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ("research/csp/csp_model.joblib", "research/csp/csp_filters_colleague.joblib")))
