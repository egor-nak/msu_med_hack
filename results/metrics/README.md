# results/metrics/

Обязательно: онлайн-симуляция на `train/*.mat`.

```bash
# на сессию -> results/metrics/per_session/<stem>.json
python solution/run.py --input <session.mat> --output-dir results/metrics/runs/<stem>
python solution/score.py --session <session.mat> \
  --predictions results/metrics/runs/<stem>/predictions.csv \
  --timings results/metrics/runs/<stem>/timings.csv \
  --output results/metrics/per_session/<stem>.json

# агрегат: невзвешенное среднее по сессиям (без суммирования G)
python solution/score.py --aggregate results/metrics/per_session/*.json \
  --output results/metrics/metrics.json
```
