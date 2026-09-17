"""
예측 서비스 (MSA).

피처를 받아 SOH/RUL 예측값만 반환한다. 모델 재학습·배포 주기가
설명 서비스와 달라 독립적으로 배포할 수 있게 분리했다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model.train import FEATURE_COLUMNS

MODEL_PATH = Path("data/processed/soh_model.joblib")
RUL_MODEL_PATH = Path("data/processed/rul_model.joblib")

app = FastAPI(title="Battery Predict Service", version="0.1.0")

_soh_model = None
_rul_model = None


def _load(path: Path):
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"모델이 없습니다: {path}")
    return joblib.load(path)


class CycleFeatures(BaseModel):
    cycles_seen: int
    ambient_temperature_c: float
    discharge_duration_s: float
    voltage_mean: float
    voltage_min: float
    voltage_std: float
    voltage_slope: float
    time_to_knee_voltage_s: float
    current_mean: float
    current_std: float
    temperature_mean: float
    temperature_max: float


class SohPrediction(BaseModel):
    predicted_soh: float


class RulPrediction(BaseModel):
    predicted_rul_cycles: float


@app.get("/health")
def health():
    return {"status": "ok", "soh_model_loaded": MODEL_PATH.exists(), "rul_model_loaded": RUL_MODEL_PATH.exists()}


@app.post("/v1/predict/soh", response_model=SohPrediction)
def predict_soh(features: CycleFeatures):
    global _soh_model
    if _soh_model is None:
        _soh_model = _load(MODEL_PATH)
    row = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]
    return SohPrediction(predicted_soh=float(_soh_model.predict(row)[0]))


@app.post("/v1/predict/rul", response_model=RulPrediction)
def predict_rul(features: CycleFeatures):
    global _rul_model
    if _rul_model is None:
        _rul_model = _load(RUL_MODEL_PATH)
    row = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS]
    return RulPrediction(predicted_rul_cycles=float(_rul_model.predict(row)[0]))
