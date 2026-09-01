from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

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


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_diagnose_normal_reading():
    resp = client.post("/v1/battery/diagnose", json=SAMPLE_READING)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["predicted_soh"] <= 1.2
    assert "explanation" in body and len(body["explanation"]) > 0


def test_diagnose_rejects_missing_fields():
    resp = client.post("/v1/battery/diagnose", json={"cycles_seen": 10})
    assert resp.status_code == 422


def test_predict_rul():
    resp = client.post("/v1/battery/predict-rul", json=SAMPLE_READING)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_rul_cycles"] >= 0
    assert "note" in body
