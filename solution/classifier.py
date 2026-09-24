"""OnlineClassifier — тонкая обёртка над hbci.HybridDecoder (ЭЭГ + NIRS, покой / левая / правая кисть).

Вызов каркасом: __init__(meta) -> load(artifacts_dir)? -> (push | predict | fit_block)*
Классы: 1 = покой, 2 = левая кисть, 3 = правая кисть. Метки — только в fit_block (прошедшие блоки).

Конфигурация: hbci.config.DEFAULTS + artifacts/config.json (финальная); OC_CFG (JSON) — только
для экспериментов. Глобальные компоненты: artifacts/global_eeg.joblib и fusion.json;
HBCI_LOSO_DIR — только для LOSO-экспериментов. HBCI_DUMP_DIR — дамп вероятностей (эксперименты).
На проверке ни одна из переменных окружения не задаётся.

Причинность: только SOS-фильтры с сохранённым состоянием; центры/нормировки/маски — по прошлому;
сглаживание — EMA по числу вызовов predict (без Δt); ЭЭГ-признаки ≤ 30 Гц; файлы сессии не читаются.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hbci.config import Config
from hbci.pipeline import HybridDecoder

HERE = Path(__file__).resolve().parent


class OnlineClassifier:
    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self._dec: HybridDecoder | None = None

    def _ensure(self, artifacts_dir: str | Path | None = None) -> HybridDecoder:
        if self._dec is None:
            ad = Path(artifacts_dir) if artifacts_dir is not None else HERE / "artifacts"
            self._dec = HybridDecoder(self.meta, Config.build(ad))
            self._dec.load(ad)
        return self._dec

    def load(self, artifacts_dir: str | Path) -> None:
        self._dec = None
        self._ensure(artifacts_dir)

    def push(self, eeg_chunk, nirs_chunk) -> None:
        self._ensure().push(eeg_chunk, nirs_chunk)

    def predict(self) -> dict:
        return self._ensure().predict()

    def fit_block(self, block_idx: int, labels: dict) -> None:
        self._ensure().fit_block(block_idx, labels)
