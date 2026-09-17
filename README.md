# Battery SOH Diagnosis Pipeline

리튬이온 배터리의 충방전 기록으로 **SOH(State of Health, 잔존 성능)** 와 **RUL(Remaining Useful Life, 잔여 수명)** 을 예측하고, 진단 결과를 API와 자연어 설명으로 제공하는 엔드투엔드 파이프라인.

> 배터리 지식 없이 읽을 수 있는 요약은 [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md), 실험 과정은 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)에 정리했다.

## 문제 정의

배터리는 충방전을 반복할수록 사용 가능한 용량이 줄고, 열화 속도는 배터리와 사용 조건마다 다르다. 지금까지의 충방전 패턴만으로 현재 성능과 교체 시점을 예측하면 정비 비용과 안전 위험을 미리 관리할 수 있다.

## 데이터

- **학습:** [NASA PCoE Battery Data Set](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) B0005 · B0006 · B0007 · B0018
  - 사이클마다 전압·전류·온도 시계열과 실측 용량(Ah)이 기록돼 있다.
  - SOH = 실측 용량 / 정격 용량(2.0Ah)
- **외부 검증:** CALCE(메릴랜드대) CS2_35~38 (LiCoO2, 각형, 정격 1.1Ah). 학습에 쓰지 않았다.

## 파이프라인

```
data/raw/*.mat (NASA PCoE)
   │  src/pipeline/parse_mat.py
   ▼
battery_cycles.csv   사이클 단위 피처 + SOH/RUL 라벨
   │  src/model/train.py, train_rul.py
   ▼
soh_model.joblib     RandomForestRegressor
   │  src/api/main.py + explain.py
   ▼
POST /v1/battery/diagnose     → { predicted_soh, soh_percent, is_below_eol_threshold, explanation }
POST /v1/battery/predict-rul  → { predicted_rul_cycles, note }
```

## 주요 설계 판단

- **배터리 단위 검증(LOBO):** 사이클을 무작위로 나누면 같은 배터리의 인접 사이클끼리 패턴이 거의 같아 성능이 부풀려진다(데이터 누수). 배터리 하나를 통째로 빼고 평가하는 Leave-One-Battery-Out으로 바꿨다.
- **LLM은 설명에만 사용:** SOH 예측과 교체 필요 여부 판단은 모델과 규칙이 맡고, LLM은 결과를 문장으로 풀어 주기만 한다. `ANTHROPIC_API_KEY`가 없거나 호출이 실패하면 규칙 기반 설명으로 대체한다(fail-open).

## 결과

### SOH 예측 (LOBO)

방전 곡선의 형태 피처(전압 표준편차, 전압-시간 기울기, 3.0V 도달 시간 = knee point)를 추가했다. 내부저항이 커질수록 knee point가 앞당겨지므로 열화를 더 직접적으로 반영할 것이라 봤다.

| 홀드아웃 | 기존 MAE / R² | 형태 피처 추가 후 |
|---|---|---|
| B0005 | 0.023 / 0.916 | 0.003 / 0.998 |
| B0006 | 0.023 / 0.934 | 0.016 / 0.961 |
| B0007 | 0.019 / 0.925 | 0.010 / 0.973 |
| B0018 | 0.017 / 0.921 | 0.003 / 0.997 |
| **평균** | **0.021 / 0.924** | **0.008 / 0.982** |

### RUL 예측 (LOBO, 단위: 사이클)

| 홀드아웃 | 기존 MAE / R² | 형태 피처 추가 후 |
|---|---|---|
| B0005 | 12.7 / 0.871 | 7.1 / 0.934 |
| B0006 | 7.2 / 0.909 | 13.7 / 0.576 |
| B0007 | 39.9 / 0.244 | 34.7 / 0.374 |
| B0018 | 14.6 / 0.727 | 6.1 / 0.941 |
| **평균** | **18.6 / 0.688** | **15.4 / 0.706** |

평균은 좋아졌지만 B0006은 나빠졌고, B0007은 여전히 예측이 어렵다. 그래서 RUL 응답에는 "참고용 추정치"라는 안내를 함께 반환한다.

### 외부 데이터셋(CALCE) 검증

NASA 4개로 학습한 최종 모델을 재학습 없이 CALCE 셀에 적용했다. CALCE에는 온도 측정치가 없어 온도 피처 3개는 25°C 가정값으로 채웠다.

| CALCE 셀 | MAE | R² | 실제 평균 SOH | 예측 평균 SOH |
|---|---|---|---|---|
| CS2_35 | 0.046 | 0.749 | 0.768 | 0.813 |
| CS2_36 | 0.069 | 0.654 | 0.739 | 0.806 |
| CS2_37 | 0.056 | 0.669 | 0.749 | 0.804 |
| CS2_38 | 0.050 | 0.722 | 0.759 | 0.807 |
| **전체** | **0.055** | **0.693** | | |

- 내부 검증(R² 0.982) 대비 R² 0.693으로 떨어졌고, 4개 셀 모두 SOH를 약 0.05 과대평가했다.
- 모델이 `time_to_knee_voltage_s` 하나에 **98.98%** 의존하고 있었다. 온도 피처 중요도는 0이라 가정값의 영향은 없었다.
- knee 피처는 두 데이터셋 모두에서 SOH와 거의 완벽히 상관되지만(0.999 / 0.9998), 값과 SOH의 대응 관계가 배터리 종류에 따라 달라 편향이 생긴다.
- 원인 분석과 추가 실험(정규화 피처, 혼합 학습)은 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)에 있다.

## MSA 변형

`src/api/main.py`는 예측과 설명을 한 프로세스에서 처리한다. 서비스로 나눈다면 어디서 경계를 둘지 보여 주려고 `src/msa/`에 세 서비스로 분리한 버전을 추가했다.

```
src/msa/
  predict_service/  피처 → SOH/RUL 예측값
  explain_service/  예측값 → 자연어 설명
  gateway/          두 서비스를 호출해 기존 API와 같은 응답 조립
```

- 분리 기준은 **배포 주기**다. 모델 재학습과 설명 문구·프롬프트 수정은 주기가 다르다.
- 설명 서비스가 실패해도 게이트웨이는 예측 결과를 반환한다.
- `tests/test_msa.py`에서 세 앱을 프로세스 안에서 연결해 호출 순서와 장애 처리를 검증했다. `docker-compose`로 컨테이너를 띄운 검증은 아직 하지 않았다.

## 실행

```bash
pip install -r requirements.txt

# 데이터 준비 → 학습
bash scripts/download_data.sh
python src/pipeline/parse_mat.py
python src/model/train.py
python -m src.model.train_rul

# API 서버 (모놀리식)
uvicorn src.api.main:app --reload

# 또는 MSA 변형
docker-compose up --build

# 테스트
pytest

# 외부 데이터셋 검증
bash scripts/download_calce.sh
python -m src.pipeline.parse_calce
python -m src.model.eval_external
```

## CI

GitHub Actions에서 두 job을 실행한다.

- `test`: 데이터 다운로드 → 파싱 → SOH/RUL 학습 → 테스트
- `external-holdout`: 최종 모델로 CALCE 외부 검증과 혼합 학습 실험을 재현

## 한계와 남은 과제

- 배터리 4개(NASA)·셀 4개(CALCE)뿐이라 폴드 수가 적고, 평균 수치가 우연일 가능성이 남아 있다.
- 처음 보는 배터리 종류에 대한 일반화(R² 0.693)를 높일 방법이 필요하다. 초기 몇 사이클로 개체차를 보정하는 방식을 검토할 예정이다.
- RUL에서 B0006이 나빠진 원인과 B0007이 어려운 원인을 곡선 형태 분석으로 더 확인해야 한다.
- `docker-compose` 실행 검증
