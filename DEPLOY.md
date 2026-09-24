# DEPLOY.md

## Окружение и версии

- Python **3.11** (проверено на CPython 3.11.15; ТЗ: 3.10–3.11). CPU, без GPU.
- Runtime-зависимости (`requirements.txt`, пины): numpy==2.2.6, scipy==1.15.3, pandas==2.3.3,
  scikit-learn==1.7.2, joblib==1.6.0. `solution/classifier.py` и `solution/hbci/` используют только numpy, scipy,
  scikit-learn, joblib.
- Анализ, фигуры, ноутбуки (`requirements-dev.txt`): + matplotlib==3.9.2, tabulate==0.9.0, nbformat==5.11.1,
  nbconvert==7.17.1, ipykernel==7.3.0.

```bash
python3.11 -m venv .venv && . .venv/bin/activate      # или: uv venv .venv --python 3.11
pip install -r requirements.txt                        # только запуск решения
pip install -r requirements-dev.txt                    # + анализ/фигуры/ноутбуки
```

## Данные

Публичный бакет организаторов: https://storage.yandexcloud.net/neuralinterfaces-train/ (`Data/*.mat`, 56 сессий).
```bash
bash tools/download_data.sh          # -> data/train/*.mat, data/Montage.mat
```

## Запуск (одна сессия) — то, что делает судья

```bash
python solution/run.py --input <session.mat> --output-dir <out> --artifacts solution/artifacts
python solution/score.py --session <session.mat> \
  --predictions <out>/predictions.csv --timings <out>/timings.csv --output <out>/metrics.json
```
`solution/artifacts/` (в git): `config.json` — финальная конфигурация, `global_eeg.joblib` — межсубъектная
ЭЭГ-модель (numpy-коэффициенты, обучена на всех 14 субъектах train), `fusion.json` — глобальные веса фьюжна.
Переменные окружения `OC_CFG`, `HBCI_LOSO_DIR`, `HBCI_DUMP_DIR` используются только в экспериментах;
на проверке не задаются.

## Каталог сессий (train или закрытый бакет)

```bash
python tools/run_all.py --data <DATA_DIR> --out results/metrics -j 1   # -j 1: честные тайминги
# длинный прогон частями: --shard 0/3, 1/3, 2/3, затем --aggregate-only
```
Результат: `results/metrics/metrics.json` (невзвешенное среднее по сессиям), `per_session/*.json`
(матрицы ошибок по сессиям), `summary.csv`, `runs/<сессия>/predictions.csv|timings.csv`.

## Обучение артефактов (офлайн, судья не запускает)

```bash
# финальная глобальная ЭЭГ-модель по всем сессиям train → solution/artifacts/global_eeg.joblib
python solution/train.py --mode all  --data data/train --out solution/artifacts
# LOSO-модели для честной оценки (эксперименты) → <out>/loso/<SUBJ>/global_eeg.joblib
python solution/train.py --mode loso --data data/train --out results/artifacts_exp/g_final
```
`train.py` берёт конфигурацию из `solution/artifacts/config.json` (или `--cfg '<json>'`) и прогоняет каждую
сессию через те же причинные фильтры, что и онлайн. `fusion.json` — см. `tools/fit_fusion.py`
(в финальной конфигурации веса фиксированы: w_eeg = 0.8, глобальные смещения 0).

Переключение набора ЭЭГ-каналов (решение команды): `solution/artifacts/config.json` → `eeg.channels`
(`sensorimotor11` по умолчанию или `all17`), затем переобучить `global_eeg.joblib` командой выше.

## Тесты

```bash
bash tests/run_all_tests.sh .venv/bin/python     # причинность, инвариантность к push, изоляция меток,
                                                 # особые сессии, тайминги, LOSO-артефакты
python tools/run_invariance_all.py              # инвариантность на всех 56 сессиях → results/tests.json
```

## Эксперименты и отчёт

```bash
python tools/run_experiments.py --exp-id <ID> --grid tools/grids/<ID>.json [--loso DIR] -j 8
python tools/collect_summary.py                 # → results/summary.md
python tools/make_figures.py                    # → results/figures/*.png
```
Подробности — `report/REPORT.md`, хронология с числами — `research/WORKLOG.md`.

## Docker

```bash
DATA_DIR=/path/to/data OUT_DIR=./results/metrics \
  docker compose run --rm solve --input /data/<session>.mat --output-dir /out
```

## Команда
- название: oligomery
- участники:
