"""
배터리 SOH 진단 API (FastAPI).

예측과 진단 설명을 별도 모듈로 나눠, 서비스를 분리할 때의 경계를
그대로 드러낸다(분리 버전은 src/msa 참고).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model.train import EOL_SOH_THRESHOLD, FEATURE_COLUMNS
from src.api.explain import explain_diagnosis

MODEL_PATH = Path("data/processed/soh_model.joblib")
RUL_MODEL_PATH = Path("data/processed/rul_model.joblib")

app = FastAPI(
    title="Battery SOH Diagnosis API",
    description="충방전 사이클 통계로부터 배터리 State of Health를 추정하고 설명한다.",
    version="0.1.0",
)

_model = None
_rul_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="모델이 아직 학습되지 않았습니다. `python src/model/train.py`를 먼저 실행하세요.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_rul_model():
    global _rul_model
    if _rul_model is None:
        if not RUL_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="RUL 모델이 아직 학습되지 않았습니다. `python -m src.model.train_rul`을 먼저 실행하세요.",
            )
        _rul_model = joblib.load(RUL_MODEL_PATH)
    return _rul_model


class CycleReading(BaseModel):
    cycles_seen: int = Field(..., description="지금까지 누적된 방전 사이클 수")
    ambient_temperature_c: float = Field(..., description="주변 온도(°C)")
    discharge_duration_s: float = Field(..., description="방전 1회 소요 시간(초)")
    voltage_mean: float
    voltage_min: float
    voltage_std: float = Field(..., description="방전 구간 전압의 표준편차")
    voltage_slope: float = Field(..., description="방전 곡선의 평균 기울기(V/s), 선형회귀 근사")
    time_to_knee_voltage_s: float = Field(
        ..., description=f"전압이 {3.0}V 이하로 처음 떨어지기까지 걸린 시간(초)"
    )
    current_mean: float
    current_std: float = Field(..., description="방전 구간 전류의 표준편차")
    temperature_mean: float
    temperature_max: float


class DiagnosisResponse(BaseModel):
    predicted_soh: float
    soh_percent: float
    is_below_eol_threshold: bool
    eol_threshold: float
    explanation: str


class RulResponse(BaseModel):
    predicted_rul_cycles: float
    note: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "soh_model_loaded": MODEL_PATH.exists(),
        "rul_model_loaded": RUL_MODEL_PATH.exists(),
    }


@app.post("/v1/battery/diagnose", response_model=DiagnosisResponse)
def diagnose(reading: CycleReading):
    model = get_model()
    row = pd.DataFrame([reading.model_dump()])[FEATURE_COLUMNS]
    predicted_soh = float(model.predict(row)[0])

    return DiagnosisResponse(
        predicted_soh=round(predicted_soh, 4),
        soh_percent=round(predicted_soh * 100, 1),
        is_below_eol_threshold=predicted_soh < EOL_SOH_THRESHOLD,
        eol_threshold=EOL_SOH_THRESHOLD,
        explanation=explain_diagnosis(reading.model_dump(), predicted_soh),
    )


@app.post("/v1/battery/predict-rul", response_model=RulResponse)
def predict_rul(reading: CycleReading):
    """잔여 수명(RUL) 예측.

    주의: 검증 결과 배터리에 따라 예측 편차가 컸다(README 참고, 홀드아웃
    LOBO 평균 R2 0.69, 배터리별로 0.24~0.91까지 편차). SOH 예측보다
    신뢰도가 낮으므로 이 값은 단독 의사결정 근거로 쓰지 말고 참고용으로만
    노출한다.
    """
    model = get_rul_model()
    row = pd.DataFrame([reading.model_dump()])[FEATURE_COLUMNS]
    predicted_rul = float(model.predict(row)[0])

    return RulResponse(
        predicted_rul_cycles=round(predicted_rul, 1),
        note="참고용 추정치입니다. 배터리별 편차가 커 SOH 예측만큼 신뢰하지 마세요.",
    )
