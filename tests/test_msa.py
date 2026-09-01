"""
MSA(예측 서비스 / 설명 서비스 / 게이트웨이) 통합 테스트.

이 프로젝트를 만든 환경에는 Docker 데몬이 없어서 docker-compose로 실제
컨테이너 3개를 띄우는 것까지는 검증하지 못했다. 대신 httpx의
ASGITransport로 각 FastAPI 앱을 프로세스 안에서 직접 연결해,
"게이트웨이가 예측 서비스 → 설명 서비스 순서로 호출해서 올바른 응답을
조립하는지"는 네트워크 없이도 확인할 수 있다.

즉 이 테스트가 보장하는 것은 서비스 간 계약(요청/응답 스키마)과
오케스트레이션 로직이지, docker-compose 네트워킹 자체는 아니다.
docker-compose 실행 자체는 README에 적어둔 대로 직접 실행해서
확인해야 한다.
"""

from __future__ import annotations

import httpx
import pytest

from src.msa.explain_service.main import app as explain_app
from src.msa.gateway import main as gateway_main
from src.msa.predict_service.main import app as predict_app

SAMPLE_READING = {
    "cycles_seen": 50,
    "ambient_temperature_c": 24,
    "discharge_duration_s": 3200,
    "voltage_mean": 3.6,
    "voltage_min": 2.7,
    "voltage_std": 0.22,
    "voltage_slope": -0.00022,
    "time_to_knee_voltage_s": 3100.0,
    "current_mean": -2.0,
    "current_std": 0.4,
    "temperature_mean": 33.0,
    "temperature_max": 41.0,
}


def _wire_gateway_to_in_process_apps(monkeypatch):
    """게이트웨이가 실제 네트워크 대신 인프로세스 ASGI 앱을 호출하도록 연결."""

    def _predict_client():
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=predict_app), base_url="http://predict"
        )

    def _explain_client():
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=explain_app), base_url="http://explain"
        )

    monkeypatch.setattr(gateway_main, "get_predict_client", _predict_client)
    monkeypatch.setattr(gateway_main, "get_explain_client", _explain_client)


@pytest.mark.asyncio
async def test_predict_service_soh():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=predict_app), base_url="http://predict"
    ) as client:
        resp = await client.post("/v1/predict/soh", json=SAMPLE_READING)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["predicted_soh"] <= 1.2


@pytest.mark.asyncio
async def test_explain_service():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=explain_app), base_url="http://explain"
    ) as client:
        resp = await client.post(
            "/v1/explain", json={"reading": SAMPLE_READING, "predicted_soh": 0.5}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_below_eol_threshold"] is True
    assert len(body["explanation"]) > 0


@pytest.mark.asyncio
async def test_gateway_orchestrates_predict_then_explain(monkeypatch):
    _wire_gateway_to_in_process_apps(monkeypatch)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_main.app), base_url="http://gateway"
    ) as client:
        resp = await client.post("/v1/battery/diagnose", json=SAMPLE_READING)

    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["predicted_soh"] <= 1.2
    assert len(body["explanation"]) > 0
    # 게이트웨이 응답 스키마가 기존 모놀리식 API(src/api/main.py)와 동일해야
    # 클라이언트가 뒷단 구조 변화를 몰라도 되게 하는 목적을 달성한다.
    assert set(body.keys()) == {
        "predicted_soh",
        "soh_percent",
        "is_below_eol_threshold",
        "eol_threshold",
        "explanation",
    }


@pytest.mark.asyncio
async def test_gateway_fails_open_when_explain_service_down(monkeypatch):
    """설명 서비스가 죽어도 진단 자체는 내려줘야 한다 (서비스 경계에서의 fail-open)."""

    def _predict_client():
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=predict_app), base_url="http://predict"
        )

    def _broken_explain_client():
        # 실제로 아무도 듣고 있지 않은 로컬 포트로 향하게 해서 연결 실패를 재현한다.
        return httpx.AsyncClient(base_url="http://127.0.0.1:1", timeout=1.0)

    monkeypatch.setattr(gateway_main, "get_predict_client", _predict_client)
    monkeypatch.setattr(gateway_main, "get_explain_client", _broken_explain_client)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_main.app), base_url="http://gateway"
    ) as client:
        resp = await client.post("/v1/battery/diagnose", json=SAMPLE_READING)

    assert resp.status_code == 200
    assert "설명 서비스에 연결할 수 없어" in resp.json()["explanation"]
