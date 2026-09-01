"""
예측 서비스 (MSA 변형).

원래 `src/api/main.py`는 모델 로딩과 예측, 설명 생성을 한 프로세스
안에서 다 처리하는 모놀리식 구조다. 포트폴리오 프로젝트 규모에서는
그게 더 단순하고 옳은 선택이지만, "실제 서비스라면 어디서 쪼갤
것인가"를 보여주기 위해 같은 로직을 두 개의 독립 서비스로 나눠봤다.

이 서비스의 책임은 딱 하나, 숫자 피처를 받아 SOH/RUL 예측값(숫자)만
돌려주는 것이다. 자연어 설명은 이 서비스가 전혀 모른다 — 그건
`explain_service`의 책임이다. 이렇게 나누면 예측 모델을 재학습해서
배포하는 주기와, 설명 문구/LLM 프롬프트를 바꾸는 주기가 서로 달라도
서로의 배포에 영향을 주지 않는다는 이점이 생긴다.
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
