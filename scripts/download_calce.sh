#!/usr/bin/env bash
# CALCE(메릴랜드대) CS2 시리즈 배터리 열화 데이터를 내려받는다.
#
# 원본은 CALCE Battery Group이 배포하는 Arbin 테스터 엑셀 원본이고,
# 여기서는 그 원본을 그대로 재배포하는 공개 미러(GitHub)에서 받는다.
# NASA PCoE 데이터와 달리 이 배터리들은 학습에 전혀 쓰이지 않은
# "진짜 새 배터리" 홀드아웃 검증용이다 (src/pipeline/parse_calce.py 참고).
set -euo pipefail

EXTERNAL_DIR="$(dirname "$0")/../data/external/calce"
mkdir -p "$EXTERNAL_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone --depth 1 \
  https://github.com/XiuzeZhou/CALCE.git \
  "$TMP_DIR/repo"

for cell in CS2_35 CS2_36 CS2_37 CS2_38; do
  cp -r "$TMP_DIR/repo/dataset/$cell" "$EXTERNAL_DIR/$cell"
done

echo "다운로드 완료 -> $EXTERNAL_DIR"
find "$EXTERNAL_DIR" -name "*.xlsx" | wc -l
echo "개 엑셀 파일 준비됨"
