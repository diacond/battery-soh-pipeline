"""
진단 결과 설명 레이어.

예측값(숫자)을 자연어 진단 코멘트로 바꾼다.

설계 원칙
  1. LLM은 필수 경로가 아니다. ANTHROPIC_API_KEY가 없거나 호출이 실패하면
     규칙 기반 설명으로 응답한다(fail-open).
  2. 위험 여부(임계값 비교)는 결정론적 규칙이 판단하고, LLM은 설명만 맡는다.
     환각이 안전 판단에 영향을 주지 않게 하기 위해서다.
"""

from __future__ import annotations

import os

from src.model.train import EOL_SOH_THRESHOLD

_ANTHROPIC_MODEL = "claude-sonnet-4-5"


def _rule_based_explanation(reading: dict, predicted_soh: float) -> str:
    pct = predicted_soh * 100
    temp = reading.get("temperature_max", reading.get("temperature_mean"))

    if predicted_soh < EOL_SOH_THRESHOLD:
        risk = (
            f"추정 SOH {pct:.1f}%로 수명 종료 기준({EOL_SOH_THRESHOLD*100:.0f}%) 미만입니다. "
            "배터리 교체 또는 정밀 점검을 권장합니다."
        )
    elif predicted_soh < EOL_SOH_THRESHOLD + 0.1:
        risk = f"추정 SOH {pct:.1f}%로 수명 종료 기준에 근접하고 있습니다. 열화 추이를 주기적으로 모니터링하세요."
    else:
        risk = f"추정 SOH {pct:.1f}%로 정상 범위입니다."

    temp_note = ""
    if temp is not None and temp > 40:
        temp_note = f" 최고 온도 {temp:.1f}°C로 다소 높게 관측되어 열화 가속 요인이 될 수 있습니다."

    return risk + temp_note


def _llm_explanation(reading: dict, predicted_soh: float) -> str | None:
    """ANTHROPIC_API_KEY가 설정된 경우에만 시도. 실패하면 None을 반환해 규칙 기반으로 폴백."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "당신은 배터리 진단 시스템의 설명 보조 에이전트입니다. "
            "아래 예측된 SOH(State of Health)와 원본 측정치를 바탕으로, "
            "정비 담당자가 바로 이해할 수 있는 2문장 이내의 한국어 진단 코멘트를 작성하세요. "
            "숫자를 새로 지어내지 말고, 주어진 값만 근거로 설명하세요.\n\n"
            f"예측 SOH: {predicted_soh:.3f} (임계값 {EOL_SOH_THRESHOLD})\n"
            f"측정치: {reading}"
        )
        resp = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return None


def explain_diagnosis(reading: dict, predicted_soh: float) -> str:
    llm_text = _llm_explanation(reading, predicted_soh)
    return llm_text if llm_text else _rule_based_explanation(reading, predicted_soh)
