"""
설명 서비스 (MSA 변형).

예측값(숫자)을 자연어 진단 코멘트로 바꾸는 책임만 가진다. 예측 모델이
뭘로 바뀌든 이 서비스는 몰라도 되고, 반대로 LLM 프롬프트나 설명 톤을
바꿔도 예측 서비스는 재배포할 필요가 없다 — 이게 이 경계를 나눈 이유다.

기존 `src/api/explain.py`의 규칙(숫자 판단은 결정론적으로, LLM은 설명만,
실패 시 fail-open)을 그대로 재사용한다.
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
