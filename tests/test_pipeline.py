from pathlib import Path

from src.pipeline.parse_mat import build_dataset
from src.model.train import compute_rul_labels, leave_one_battery_out_eval


def test_build_dataset_produces_expected_batteries(tmp_path):
    raw_dir = Path("data/raw")
    out_path = tmp_path / "battery_cycles.csv"
    df = build_dataset(raw_dir, out_path)

    assert out_path.exists()
    assert set(df["battery_id"].unique()) == {"B0005", "B0006", "B0007", "B0018"}
    assert df["soh"].between(0, 1.2).all()


def test_rul_labels_are_non_negative(tmp_path):
    raw_dir = Path("data/raw")
    df = build_dataset(raw_dir, tmp_path / "battery_cycles.csv")
    labeled = compute_rul_labels(df)
    assert (labeled["rul_cycles"] >= 0).all()


def test_lobo_eval_returns_one_result_per_battery(tmp_path):
    raw_dir = Path("data/raw")
    df = build_dataset(raw_dir, tmp_path / "battery_cycles.csv")
    labeled = compute_rul_labels(df)
    results = leave_one_battery_out_eval(labeled)
    assert {r["held_out_battery"] for r in results} == {"B0005", "B0006", "B0007", "B0018"}
    assert all(r["mae"] < 0.1 for r in results)
