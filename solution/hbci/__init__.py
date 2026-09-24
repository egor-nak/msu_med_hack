"""hbci — причинный онлайн-декодер ЭЭГ+NIRS (покой / левая / правая кисть).

Модули:
  config     — конфигурация (дефолты, пресеты, merge с artifacts/config.json и OC_CFG);
  filters    — потоковые SOS-фильтры с сохранённым состоянием;
  buffers    — растущие массивы, NIRS-таймстемпы;
  riemann    — операции над SPD-матрицами (numpy/eigh), средние, касательное пространство;
  eeg        — фронтенд ЭЭГ: ВЧ-фильтр, референс, банк полос, ковариации окон на сетке 62 отсчёта;
  nirs       — фронтенд NIRS: фильтры, базлайн, уровни/наклоны/лаги, CBSI, GSR;
  models     — линейные модели (sLDA/LR) в виде numpy-коэффициентов, MDM, CSP, глобальная модель;
  fusion     — слияние модальностей и калибровка смещений под min-F1;
  smoothing  — EMA по числу вызовов predict (без Δt);
  pipeline   — HybridDecoder: связывает всё (push / predict / fit_block).
"""

from .pipeline import HybridDecoder  # noqa: F401
