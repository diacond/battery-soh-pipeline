"""
설명 서비스 (MSA).

예측값을 자연어 진단 코멘트로 바꾸는 역할만 맡는다. 설명 문구나 LLM
프롬프트를 바꿔도 예측 서비스를 다시 배포할 필요가 없다.
규칙은 src/api/explain.py를 그대로 재사용한다.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.api.explain import explain_diagnosis
from src.model.train import EOL_SOH_THRESHOLD

app = FastAPI(title="Battery Explain Service", version="0.1.0")


class ExplainRequest(BaseModel):
    reading: dict
    predicted_soh: float


class ExplainResponse(BaseModel):
    explanation: str
    is_below_eol_threshold: bool
    eol_threshold: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    return ExplainResponse(
        explanation=explain_diagnosis(req.reading, req.predicted_soh),
        is_below_eol_threshold=req.predicted_soh < EOL_SOH_THRESHOLD,
        eol_threshold=EOL_SOH_THRESHOLD,
    )
