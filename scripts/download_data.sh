#!/usr/bin/env bash
# NASA Ames Prognostics Center of Excellence (PCoE) Battery Data Set을 내려받는다.
# 원본 배포처: NASA PCoE Data Set Repository
#   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
# 이 스크립트는 원본 .mat 파일을 그대로 배포하는 공개 미러(GitHub)에서 받는다.
set -euo pipefail

RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone --depth 1 \
  https://github.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset.git \
  "$TMP_DIR/repo"

cp "$TMP_DIR"/repo/B*.mat "$RAW_DIR/"
echo "다운로드 완료 -> $RAW_DIR"
ls -la "$RAW_DIR"
