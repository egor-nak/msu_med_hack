# Гибридный ИМК ЭЭГ+NIRS

Шаблон репозитория команды (SourceCraft). Трек «Будущее медицины».
РНИМУ им. Н. И. Пирогова, отдел нейрокомпьютерных интерфейсов.

**ТЗ:** [`docs/TZ.md`](docs/TZ.md) ([`TZ.docx`](docs/TZ.docx) с рисунками) · **данные:** [`docs/DATASET.md`](docs/DATASET.md)

## Решение команды (кратко)

Причинный гибридный декодер `solution/hbci/`: римановы признаки ЭЭГ (касательное пространство + MDM, 4–30 Гц,
сенсомоторные каналы) с межсубъектной моделью и перецентрированием, NIRS-наклоны, лог-линейный фьюжн,
калибровка смещений под min-F1. LOSO на 56 сессиях train: $F_{global}$ = 0.507, $P_{global}$ = 0.640,
NIRS min-F1 = 0.403, балл 60.65 без штрафов (MVP: 57.58, из них ≈ 5.5 балла — утечка тайминга).
Отчёт — [`report/REPORT.md`](report/REPORT.md), сводка экспериментов — [`results/summary.md`](results/summary.md),
фигуры — [`results/figures/`](results/figures/), развёртывание — [`DEPLOY.md`](DEPLOY.md).

## Задача

Онлайн-пригодный классификатор по синхронным ЭЭГ+NIRS на 3 класса:

| Класс | Состояние |
|-------|-----------|
| 1 | Покой |
| 2 | Представление раскрытия левой кисти |
| 3 | Представление раскрытия правой кисти |

Окно **1 с**, шаг **250 мс**, без заглядывания в будущее; дообучение после завершённых блоков; блок 1 не оценивается.

## Данные (кратко)

56 сессий `.mat`, 14 испытуемых (путь к бакету сообщит организатор). Важно:
у части сессий нет NIRS;

Подробности — в [`docs/DATASET.md`](docs/DATASET.md). Необязательный пример чтения файла — [`example.py`](example.py) (можно использовать, можно игнорировать).

## Быстрый старт

```bash
pip install -r requirements.txt

# опционально: посмотреть поля одной сессии
python example.py /path/to/session.mat

python solution/run.py --input /path/to/session.mat --output-dir results/metrics/run1
python solution/score.py --session /path/to/session.mat \
  --predictions results/metrics/run1/predictions.csv \
  --timings results/metrics/run1/timings.csv \
  --output results/metrics/run1/metrics.json
```

Docker: `docker compose run --rm solve --input /data/session.mat --output-dir /out`

## Контракт

Реализуйте `OnlineClassifier` в [`solution/classifier.py`](solution/classifier.py):

```python
def push(self, eeg_chunk, nirs_chunk): ...
def predict(self) -> dict:  # {"y": 1|2|3, "y_nirs"?}
def fit_block(self, block_idx, labels): ...
```

`run.py` / `score.py` / `io_utils.py` **заморожены**. Ядро алгоритма — любой язык через эту обёртку (ТЗ §3.2).

Опционально: `solution/train.py` (офлайн; проверка **не** запускает). Онлайн-дообучение — только `fit_block`.

## Оценка

База 130: `60×F_global + 30×P_global + NIRS≤20 + дизайн/отчёт≤20 − штрафы`.
- \(F_{\mathrm{global}}\), \(P_{\mathrm{global}}\) — **невзвешенное среднее** min-F1 и macro-recall по сессиям (не сумма G).
- NIRS-бонус: 5 за валидную матрицу `y_nirs` + `15 × F_global^NIRS` (худшая F1 NIRS). ЭЭГ-only в балл не входит.
- Штрафы: `predict` > **0.20 с**, `fit_block` > 1.5 с.

Сдача метрик: онлайн-симуляция на `train/` → `results/metrics/` (ТЗ §5.3). Полная формула — в ТЗ §5.

## Структура и сдача

```
.
├── README.md, DEPLOY.md, requirements.txt, Dockerfile, docker-compose.yml
├── example.py         необязательный пример чтения .mat
├── docs/TZ.md, docs/DATASET.md
├── solution/          classifier.py (+ замороженные run/score/io_utils), artifacts/
├── assets/            ноутбуки (.ipynb) — см. assets/README.md
├── results/metrics/   онлайн-метрики на train (обязательно)
├── presentation/      слайды (~7+7 мин); шаблон — docs/Шаблон_защита проекта.pptx
└── report/            текстовый отчёт
```

Крупные `.mat` / кэши — в S3, не в git. В git: код, ноутбуки с выводами, метрики, презентация, отчёт, пины в DEPLOY.md.

## Инфраструктура

[SourceCraft](https://sourcecraft.dev/portal/docs/en/sourcecraft/quickstart) ·
[DataSphere](https://yandex.cloud/en/docs/datasphere/) ·
[S3 Connector](https://yandex.cloud/en/docs/datasphere/operations/data/s3-connectors)
