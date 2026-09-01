from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_diagnose_normal_reading():
    sample = {
        "cycles_seen": 50,
        "ambient_temperature_c": 24,
        "discharge_duration_s": 3200,
        "voltage_mean": 3.6,
        "voltage_min": 2.7,
        "current_mean": -2.0,
        "temperature_mean": 33.0,
        "temperature_max": 41.0,
    }
    resp = client.post("/v1/battery/diagnose", json=sample)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["predicted_soh"] <= 1.2
    assert "explanation" in body and len(body["explanation"]) > 0


def test_diagnose_rejects_missing_fields():
    resp = client.post("/v1/battery/diagnose", json={"cycles_seen": 10})
    assert resp.status_code == 422
