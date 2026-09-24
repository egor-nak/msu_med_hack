#!/usr/bin/env bash
# Все тесты утечек/контракта на 3 сессиях: обычная, без NIRS, NIRS 3.9 Гц.
#   bash tests/run_all_tests.sh [python]
set -euo pipefail
PY="${1:-${PYTHON:-python}}"
if [[ "$PY" == */* ]]; then PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"; fi
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=../data/train
SESSIONS=("$D/2025.12.04.13.29.33_N01.mat" "$D/2025.12.18.12.09.22_N03.mat" "$D/2026.02.19.13.23.35_N13.mat")
for s in "${SESSIONS[@]}"; do
  echo "=== $(basename "$s")"
  "$PY" test_causal.py "$s" | tail -1
  "$PY" test_chunk_invariance.py "$s" | tail -1
  "$PY" test_label_isolation.py "$s" | tail -1
done
echo "=== особые сессии";  "$PY" test_edge_sessions.py "$D" 2>/dev/null | tail -1
echo "=== тайминги";       "$PY" test_timing.py | tail -2
echo "=== LOSO-артефакты"; "$PY" test_artifacts_loso.py | tail -1
echo "ВСЕ ТЕСТЫ ПРОЙДЕНЫ"
