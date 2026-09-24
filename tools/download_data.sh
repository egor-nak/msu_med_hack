#!/usr/bin/env bash
# Скачивает train-данные из публичного бакета организаторов в ./data/
# Использование: bash tools/download_data.sh [папка_назначения]
set -euo pipefail
BUCKET_URL="https://storage.yandexcloud.net/neuralinterfaces-train"
DEST="${1:-data}"
mkdir -p "$DEST/train"

keys=$(curl -fsS "$BUCKET_URL/?list-type=2&max-keys=1000" | grep -o '<Key>[^<]*</Key>' | sed -e 's/<Key>//' -e 's/<\/Key>//')

for key in $keys; do
  case "$key" in
    Data/*.mat) out="$DEST/train/${key#Data/}" ;;
    Montage.mat) out="$DEST/Montage.mat" ;;
    *) continue ;;
  esac
  if [ -s "$out" ]; then echo "есть: $out"; continue; fi
  echo "качаю: $key"
  curl -fsS --retry 3 -o "$out.part" "$BUCKET_URL/$key" && mv "$out.part" "$out"
done

n=$(ls "$DEST"/train/*.mat 2>/dev/null | wc -l | tr -d ' ')
echo "Готово: $n файлов в $DEST/train (ожидается 56)"
