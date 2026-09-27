"""Конфигурация декодера.

Порядок сборки конфигурации:
  1. ``DEFAULTS`` (код);
  2. если ``OC_CFG`` не задан — ``artifacts/config.json`` (финальная конфигурация сдачи);
     если ``OC_CFG`` задан — config.json читается только при ``"use_artifact_config": true``;
  3. ``"preset"`` (если указан) применяется поверх дефолтов, до остальных ключей;
  4. ``OC_CFG`` (JSON) — переопределение для экспериментов; на проверке не задаётся.

Слияние — рекурсивное по словарям; списки заменяются целиком.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Имена каналов ЭЭГ в порядке колонок EEG.Raw (см. io_utils.EEG_CHANNEL_NAMES).
CHANNEL_SETS: dict[str, list[str]] = {
    # MVP: все каналы без lpa/rpa/fp1/fp2
    "all17": ["c4", "f8", "p8", "f4", "p4", "o2", "cz", "pz", "fz", "o1", "p3", "f3", "p7", "f7", "c3", "c7", "c8"],
    "no_f7f8": ["c4", "p8", "f4", "p4", "o2", "cz", "pz", "fz", "o1", "p3", "f3", "p7", "c3", "c7", "c8"],
    "sensorimotor11": ["c3", "cz", "c4", "f3", "fz", "f4", "p3", "pz", "p4", "c7", "c8"],
    "sensorimotor9": ["c3", "cz", "c4", "f3", "fz", "f4", "p3", "pz", "p4"],
    "frontocc": ["fp1", "fp2", "f7", "f8", "o1", "o2"],
    "all21": ["c4", "rpa", "f8", "p8", "f4", "p4", "fp2", "o2", "cz", "pz", "fz", "o1", "fp1", "p3",
              "f3", "p7", "f7", "lpa", "c3", "c7", "c8"],
}

# Каналы, исключаемые из CAR («грязные»: глаза, края, затылок с клиппингом).
CAR_EXCLUDE = ["fp1", "fp2", "f7", "f8", "lpa", "rpa", "o1", "o2"]

# Малый лапласиан для C3/C4/Cz (c7/c8 по координатам — T7/T8).
LAPLACIAN = {
    "c3": ["f3", "p3", "c7", "cz"],
    "c4": ["f4", "p4", "c8", "cz"],
    "cz": ["fz", "pz", "c3", "c4"],
}

DEFAULTS: dict[str, Any] = {
    "eeg": {
        "enabled": True,
        "features": "ts",              # ts | logpower | mdm | ts+mdm | csp | heog | csponly
        "channels": "sensorimotor11",  # имя из CHANNEL_SETS или список имён
        "reference": "car_clean",      # none | car_clean | laplacian
        "hp_hz": 0.5,                  # 0 — без ВЧ-фильтра (как MVP)
        "bands": [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 30]],
        "band_order": 4,
        "cov": "oas",                  # oas | lw | scm
        "cov_eps": 1e-4,               # eps·trace/n·I
        "cov_avg_n": 1,                # усреднение ковариаций по n последним окнам сетки (1 с + (n−1)·0.248 с)
        "mean": "logeuclid",           # logeuclid | riemann (для Cref и центроидов MDM)
        "riemann_max_n": 400,          # подвыборка окон для итераций Карчера
        "classifier": "slda",          # slda | lr
        "C": 0.1,                      # для lr
        "logit_T": None,               # нормировка логитов сессионной модели (SD=1 на обучении, / T); None — выкл
        "mdm_T": 1.0,                  # температура MDM (в единицах медианы d²)
        "mdm_metric": "riemann",       # riemann (центроиды Карчера/ЛЕ, риманово расстояние) | logeuclid (в TS, быстро)
        "csp_n": 2,                    # фильтров на класс на полосу (B2)
        "train_skip_s": 0.0,
        "bad_channels": True,
        "bad_flat_frac": 0.05,         # доля «плоских» отсчётов (клиппинг/обрыв)
        "bad_ratio": 8.0,              # мощность канала / медиана по каналам > ratio — шумный контакт
        "csp": {                       # CSP + лог-мощность + асимметрия (векторизатор коллеги), hbci/csp.py
            "enabled": False,
            "file": "csp_filters.joblib",   # обученные офлайн фильтры (numpy): tools/convert_csp.py | train_csp.py
            "prep": "raw",             # вход CSP: "raw" (сырые отсчёты, как обучена CSP коллеги) | [lo, hi] Гц
            "avg_n": 1,                # усреднение CSP-ковариаций по окнам сетки
            "use_csp": True, "use_power": True, "use_asym": True,
            "symmetry": [["c3", "c4"], ["f3", "f4"], ["p3", "p4"], ["c7", "c8"]],
        },
        "heog_tau_s": 20.0,            # только для контрольного heog-декодера
        "heog_band": [0.1, 3.0],
    },
    "nirs": {
        "enabled": True,
        "features": "hemo",            # hemo | mvp
        "hp_hz": 0.01,
        "lp_hz": 0.4,
        "signal": "hb",                # hb (HbO и HbR) | cbsi | hbo
        "gsr": False,                  # регрессия глобального сигнала (коэф. по прошлому)
        "spatial": "channels",         # channels | regions (правое/левое/центр)
        "level_s": 2.0,
        "base_tau_s": 20.0,
        "slope_s": 4.0,
        "lags_s": [0, 2, 4, 6],
        "use_level": True,
        "use_slope": True,
        "lateral": True,
        "classifier": "slda",          # slda | lr | cascade
        "C": 0.1,
        "logit_T": None,               # нормировка логитов NIRS-модели (см. eeg.logit_T)
        # признаки MVP (для B1-fix): cur-base, cur-lag по ящичным средним
        "mvp_short": 2.0, "mvp_long": 20.0, "mvp_lag": 4.0,
    },
    "global": {
        "enabled": False,              # глобальная (межсубъектная) ЭЭГ-модель из artifacts
        "k": 1.0,                      # вес w = k/(k+n_blocks); "inf" — только глобальная
        "C": 0.1,                      # C логистической регрессии (train.py)
        "recenter": "block1",          # block1 — Cref по блоку 1 (как в train.py) | all — по всем прошлым окнам
        "file": "global_eeg.joblib",
    },
    "smoothing": {
        "tau_s": 2.0,                  # EMA по вызовам: a = exp(-1/(tau_s·calls_per_s)); 0 — выкл
        "nirs_tau_s": None,            # отдельная τ для NIRS-ветки (None — как tau_s)
        "calls_per_s": 4.0,
        "space": "log",                # log (лог-вероятности) | prob (как MVP)
    },
    "fusion": {
        "mode": "loglinear",           # loglinear | avg | stacking
        "w_eeg": 0.8,
        "file": "fusion.json",
        "use_file": True,              # брать веса/смещения из fusion.json, если есть
    },
    "bias": {
        "mode": "none",                # none | global | session (session = LOBO + усадка к global)
        "min_blocks": 3,
        "grid": [-1.5, 1.5, 0.1],
        "shrink_k": 3.0,               # λ = n/(n+k)
        "passes": 2,
        "fixed": [0.0, 0.0, 0.0],      # ручное смещение (добавляется всегда)
    },
    "priors": [1 / 3, 1 / 3, 1 / 3],
    "random_state": 0,
}

PRESETS: dict[str, dict[str, Any]] = {
    # MVP с двумя исправлениями (масштаб NIRS фиксируется в fit; EMA по вызовам)
    "b1fix": {
        "eeg": {"features": "logpower", "channels": "all17", "reference": "none", "hp_hz": 0.0,
                "bands": [[8, 12], [12, 16], [16, 24], [24, 30]], "bad_channels": False},
        "nirs": {"features": "mvp", "hp_hz": 0.0, "lp_hz": 0.0, "classifier": "slda"},
        "smoothing": {"tau_s": 2.0, "space": "prob"},
        "fusion": {"mode": "loglinear", "w_eeg": 0.8, "use_file": False},
        "bias": {"mode": "none"},
    },
    # контрольные «артефактные» декодеры (НЕ для финала)
    "conf_heog": {
        "eeg": {"features": "heog", "channels": "all21", "reference": "none", "hp_hz": 0.0,
                "bands": [[8, 12]], "bad_channels": False},
        "nirs": {"enabled": False},
        "smoothing": {"tau_s": 2.0},
        "fusion": {"use_file": False},
    },
    "conf_fes": {
        "eeg": {"features": "logpower", "channels": "all21", "reference": "none", "hp_hz": 0.5,
                "bands": [[80, 86]], "bad_channels": False},
        "nirs": {"enabled": False},
        "smoothing": {"tau_s": 2.0},
        "fusion": {"use_file": False},
    },
    "conf_frontocc": {
        "eeg": {"features": "logpower", "channels": "frontocc", "reference": "none", "hp_hz": 0.5,
                "bands": [[8, 12], [12, 16], [16, 24], [24, 30]], "bad_channels": False},
        "nirs": {"enabled": False},
        "smoothing": {"tau_s": 2.0},
        "fusion": {"use_file": False},
    },
    # воспроизведение пайплайна авторов: FB multiclass CSP + sLDA; NIRS-каскад; среднее вероятностей
    "b2_original": {
        "eeg": {"features": "csp", "channels": "all17", "reference": "none", "hp_hz": 0.5,
                "bands": [[4, 7], [7, 13], [13, 30]], "cov": "lw", "csp_n": 2, "bad_channels": False},
        "nirs": {"features": "hemo", "classifier": "cascade", "lags_s": [0], "lateral": False},
        "smoothing": {"tau_s": 0.0},
        "fusion": {"mode": "avg", "use_file": False},
        "bias": {"mode": "none"},
    },
}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивное слияние словарей (over поверх base); возвращает новый словарь."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply(cfg: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    over = dict(over)
    preset = over.pop("preset", None)
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"неизвестный пресет {preset!r}")
        cfg = deep_merge(cfg, PRESETS[preset])
        cfg["preset"] = preset
    return deep_merge(cfg, over)


@dataclass
class Config:
    """Собранная конфигурация: словарь + удобный доступ по секциям."""

    raw: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    @property
    def eeg(self) -> dict[str, Any]:
        return self.raw["eeg"]

    @property
    def nirs(self) -> dict[str, Any]:
        return self.raw["nirs"]

    @property
    def glob(self) -> dict[str, Any]:
        return self.raw["global"]

    @property
    def smoothing(self) -> dict[str, Any]:
        return self.raw["smoothing"]

    @property
    def fusion(self) -> dict[str, Any]:
        return self.raw["fusion"]

    @property
    def bias(self) -> dict[str, Any]:
        return self.raw["bias"]

    @classmethod
    def build(cls, artifacts_dir: str | Path | None = None, env: dict[str, str] | None = None) -> "Config":
        env = os.environ if env is None else env
        cfg = copy.deepcopy(DEFAULTS)
        oc = env.get("OC_CFG")
        over = json.loads(oc) if oc else {}
        use_art = (not oc) or bool(over.pop("use_artifact_config", False))
        if use_art and artifacts_dir is not None:
            p = Path(artifacts_dir) / "config.json"
            if p.exists():
                cfg = _apply(cfg, json.loads(p.read_text(encoding="utf-8")))
        cfg = _apply(cfg, over)
        cfg["_use_artifacts"] = use_art  # эксперименты с OC_CFG не читают финальные артефакты
        return cls(cfg)


def resolve_channels(spec: str | list[str]) -> list[str]:
    """Имя набора каналов или явный список → список имён."""
    if isinstance(spec, str):
        return list(CHANNEL_SETS[spec])
    return [str(c).lower() for c in spec]
