# Battery SOH Diagnosis Pipeline

리튬이온 배터리의 충방전 이력으로부터 State of Health(SOH, 잔존 성능)를
예측하고, 진단 결과를 실시간 API와 자연어 설명으로 제공하는 엔드투엔드
파이프라인입니다. EV/ESS 배터리의 안전성·수명·잔존 성능을 평가·인증하는
서비스를 소규모로 재현한 포트폴리오 프로젝트입니다.

## 문제 정의

배터리는 충방전을 반복할수록 실제 사용 가능한 용량이 점점 줄어듭니다.
이 열화(degradation) 속도는 배터리마다, 사용 조건(온도·부하)마다 달라서,
"지금까지의 충방전 패턴만 보고 이 배터리가 지금 몇 %짜리인지, 언제
교체 시점에 도달할지"를 예측할 수 있으면 정비 비용과 안전 리스크를
사전에 관리할 수 있습니다. 이 프로젝트는 그 예측 문제를 다룹니다.

## 데이터

[NASA Ames Prognostics Center of Excellence(PCoE) Battery Data Set](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
중 B0005, B0006, B0007, B0018 4개 배터리의 반복 충방전 이력을 사용합니다.
배터리마다 수백 회의 방전 사이클이 기록돼 있고, 사이클마다 전압·전류·
온도 시계열과 그 시점의 실측 용량(Capacity, Ah)이 포함됩니다. 정격 용량
2.0Ah 대비 실측 용량의 비율을 SOH로 정의했습니다.

## 파이프라인 구조

```
data/raw/*.mat  (원본, NASA PCoE)
      │  src/pipeline/parse_mat.py
      ▼
data/processed/battery_cycles.csv  (사이클 단위 요약 피처 + SOH + RUL 라벨)
      │  src/model/train.py
      ▼
data/processed/soh_model.joblib  (RandomForestRegressor)
      │  src/api/main.py + src/api/explain.py
      ▼
POST /v1/battery/diagnose  →  { predicted_soh, is_below_eol_threshold, explanation }
```

## 해결 과정에서의 주요 판단

**평가 방법으로 Leave-One-Battery-Out을 선택한 이유.** 처음에는 전체
사이클을 무작위로 train/test로 나눴는데, 같은 배터리의 인접 사이클끼리
패턴이 거의 동일해서 R²가 비현실적으로 높게 나왔습니다(데이터 누수).
실제 서비스에서 이 모델이 마주할 상황은 "한 번도 본 적 없는 새 배터리의
초기 사이클만 보고 미래를 예측하는 것"이므로, 배터리 단위로 통째로
홀드아웃하는 방식으로 바꿨습니다. 그 결과 각 배터리를 순서대로
홀드아웃했을 때 평균 MAE 0.021(SOH 스케일, 약 2%p), 평균 R² 0.92를
얻었습니다 — 무작위 분할보다 낮지만 훨씬 신뢰할 수 있는 수치입니다.

**LLM 사용 범위를 의도적으로 제한한 이유.** `src/api/explain.py`는
숫자 예측과 위험 여부 판단(임계값 비교)을 결정론적 규칙에 맡기고, LLM은
그 결과를 자연어로 풀어 설명하는 역할만 담당합니다. `ANTHROPIC_API_KEY`가
없거나 호출이 실패해도 규칙 기반 설명으로 자동 대체(fail-open)되어
서비스가 죽지 않습니다. 안전과 직결된 판단(교체 필요 여부)에 생성형
모델의 환각(hallucination) 리스크를 끌어들이지 않기 위한 선택입니다.

## 결과

### SOH 예측

| 홀드아웃 배터리 | 테스트 사이클 수 | MAE | R² |
|---|---|---|---|
| B0005 | 168 | 0.023 | 0.916 |
| B0006 | 168 | 0.023 | 0.934 |
| B0007 | 168 | 0.019 | 0.925 |
| B0018 | 132 | 0.017 | 0.921 |

### RUL(잔여 수명, cycles) 예측 — `src/model/train_rul.py`

SOH와 같은 피처로 "몇 사이클 후 EOL(SOH<0.7)에 도달하는가"를 별도
모델로 예측했다. SOH 예측과 달리 배터리별 편차가 크다는 게 정직한
결과다.

| 홀드아웃 배터리 | 테스트 사이클 수 | MAE(cycles) | R² |
|---|---|---|---|
| B0005 | 168 | 12.7 | 0.871 |
| B0006 | 168 | 7.2 | 0.909 |
| B0007 | 168 | 39.9 | 0.244 |
| B0018 | 132 | 14.6 | 0.727 |

B0007을 홀드아웃했을 때만 유독 R²가 낮다. B0007의 열화 곡선이 다른
3개 배터리와 형태가 달라서(다른 배터리로 학습한 패턴이 B0007에는 잘
안 맞음) 벌어지는 현상으로 보인다. 그래서 API 응답에도 RUL 예측값에는
"참고용, SOH만큼 신뢰하지 말 것"이라는 note를 같이 반환하게 만들었다 —
모델 성능을 실제보다 좋게 포장하지 않고, 이 모델을 그대로 배포한다면
사용자에게 불확실성을 정직하게 전달해야 한다고 판단했다.

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 원본 데이터 준비 (없다면)
bash scripts/download_data.sh

# 2) 파이프라인 실행
python src/pipeline/parse_mat.py
python src/model/train.py
python -m src.model.train_rul

# 3) API 서버 실행
uvicorn src.api.main:app --reload

# 4) 테스트
pytest
```

## CI

`.github/workflows/ci.yml` — push/PR마다 데이터 다운로드부터 파이프라인,
SOH/RUL 모델 학습, 테스트까지 전체 파이프라인을 처음부터 재현해서
검증한다. 단순히 "테스트만 통과하면 끝"이 아니라, 이 저장소를 클론한
사람이 문서대로 따라 했을 때 실제로 재현되는지까지 매번 확인하는 게
목적이다.

## 남은 과제

- 현재는 사이클 단위 스냅샷 피처만 쓰는데, 방전 곡선 자체(전압-시간
  곡선의 형태)를 시계열 피처로 추가하면 정확도를 더 끌어올릴 여지가 있음
  (특히 RUL 예측에서 편차가 큰 B0007 케이스)
- Docker 컨테이너화 및 MSA 경계(예측 서비스 / 설명 서비스)로 분리
- RUL 예측의 배터리 간 편차 원인을 더 파고들어(곡선 형태 클러스터링 등)
  근본 원인 설명 추가
