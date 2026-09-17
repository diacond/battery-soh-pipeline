"""
게이트웨이 (MSA).

predict_service → explain_service 순서로 호출해 모놀리식 API
(/v1/battery/diagnose)와 같은 응답을 조립한다.
내부 주소는 PREDICT_SERVICE_URL, EXPLAIN_SERVICE_URL 환경변수로 받는다
(기본값: http://predict:8001, http://explain:8002).
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PREDICT_SERVICE_URL = os.environ.get("PREDICT_SERVICE_URL", "http://predict:8001")
EXPLAIN_SERVICE_URL = os.environ.get("EXPLAIN_SERVICE_URL", "http://explain:8002")
EOL_SOH_THRESHOLD = 0.70

app = FastAPI(title="Battery Diagnosis Gateway", version="0.1.0")


def get_predict_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=PREDICT_SERVICE_URL, timeout=5.0)


def get_explain_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=EXPLAIN_SERVICE_URL, timeout=10.0)


class CycleReading(BaseModel):
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


class DiagnosisResponse(BaseModel):
    predicted_soh: float
    soh_percent: float
    is_below_eol_threshold: bool
    eol_threshold: float
    explanation: str


@app.get("/health")
async def health():
    """게이트웨이 자신뿐 아니라 뒷단 서비스들이 살아있는지도 함께 확인한다."""
    statuses = {}
    async with get_predict_client() as client:
        try:
            r = await client.get("/health")
            statuses["predict_service"] = r.json()
        except httpx.HTTPError as exc:
            statuses["predict_service"] = {"status": "unreachable", "error": str(exc)}

    async with get_explain_client() as client:
        try:
            r = await client.get("/health")
            statuses["explain_service"] = r.json()
        except httpx.HTTPError as exc:
            statuses["explain_service"] = {"status": "unreachable", "error": str(exc)}

    return {"status": "ok", "dependencies": statuses}


@app.post("/v1/battery/diagnose", response_model=DiagnosisResponse)
async def diagnose(reading: CycleReading):
    payload = reading.model_dump()

    async with get_predict_client() as client:
        try:
            resp = await client.post("/v1/predict/soh", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"predict_service 호출 실패: {exc}") from exc
    predicted_soh = resp.json()["predicted_soh"]

    async with get_explain_client() as client:
        try:
            resp = await client.post(
                "/v1/explain", json={"reading": payload, "predicted_soh": predicted_soh}
            )
            resp.raise_for_status()
            explanation = resp.json()["explanation"]
        except httpx.HTTPError:
            # 설명 서비스가 죽어도 예측 자체는 이미 확보했으니 진단 응답은 내려준다
            # (fail-open을 서비스 경계에서도 동일하게 적용).
            explanation = f"SOH {predicted_soh*100:.1f}% (설명 서비스에 연결할 수 없어 기본 수치만 표시합니다.)"

    return DiagnosisResponse(
        predicted_soh=round(predicted_soh, 4),
        soh_percent=round(predicted_soh * 100, 1),
        is_below_eol_threshold=predicted_soh < EOL_SOH_THRESHOLD,
        eol_threshold=EOL_SOH_THRESHOLD,
        explanation=explanation,
    )
