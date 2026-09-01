"""
배터리 SOH 진단 API (FastAPI).

LG에너지솔루션 Data Engineering JD의 "고객이 배터리 진단 결과를 실시간으로
확인 가능한 백엔드 서비스 및 API"를 가장 작은 단위로 구현한 것.
운영 서비스라면 MSA로 쪼개겠지만, 포트폴리오 규모에서는 단일 서비스 안에
(1) 예측 (2) 진단 설명 두 책임을 분리된 모듈로만 나눠 MSA로 쪼갤 때의
경계를 그대로 보여준다.
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

app = FastAPI(
    title="Battery SOH Diagnosis API",
    description="충방전 사이클 통계로부터 배터리 State of Health를 추정하고 설명한다.",
    version="0.1.0",
)

_model = None


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


class CycleReading(BaseModel):
    cycles_seen: int = Field(..., description="지금까지 누적된 방전 사이클 수")
    ambient_temperature_c: float = Field(..., description="주변 온도(°C)")
    discharge_duration_s: float = Field(..., description="방전 1회 소요 시간(초)")
    voltage_mean: float
    voltage_min: float
    current_mean: float
    temperature_mean: float
    temperature_max: float


class DiagnosisResponse(BaseModel):
    predicted_soh: float
    soh_percent: float
    is_below_eol_threshold: bool
    eol_threshold: float
    explanation: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


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
