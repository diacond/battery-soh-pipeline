"""
NASA Prognostics Center Battery Data Set 파서.

원본 출처: NASA Ames Research Center, Prognostics Center of Excellence (PCoE)
Battery Data Set (B0005, B0006, B0007, B0018) — 리튬이온 배터리를 반복
충방전하며 방전 용량(Capacity)이 열화되는 과정을 기록한 공개 데이터셋.

각 .mat 파일은 배터리 1개의 전체 충/방전 사이클(cycle) 시퀀스를 담고 있고,
방전(discharge) 사이클마다 그 순간의 실측 용량(Capacity, Ah)이 기록돼 있다.
이 스크립트는 방전 사이클만 골라 사이클 단위 요약 테이블로 펼친다.

State of Health(SOH)는 정격 용량(2.0Ah) 대비 실측 용량의 비율로 정의한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

RATED_CAPACITY_AH = 2.0  # NASA PCoE 데이터셋 공식 정격 용량
VOLTAGE_KNEE_THRESHOLD_V = 3.0  # 방전 말기 급격한 전압 강하가 시작되는 기준 전압


def _voltage_slope(time: np.ndarray, voltage: np.ndarray) -> float:
    """방전 곡선의 평균 기울기(V/s). 1차 선형회귀로 근사."""
    if time.size < 2:
        return np.nan
    slope, _ = np.polyfit(time, voltage, deg=1)
    return float(slope)


def _time_to_voltage_threshold(time: np.ndarray, voltage: np.ndarray, threshold: float) -> float:
    """전압이 threshold 이하로 처음 떨어지는 시점(초). 못 도달하면 전체 방전시간을 반환.

    이 값이 짧을수록 "내부저항이 커져서 전압이 빨리 무너진다"는 뜻이라
    열화가 상당히 진행된 신호로 해석할 수 있다.
    """
    if time.size == 0:
        return np.nan
    below = np.where(voltage <= threshold)[0]
    if below.size == 0:
        return float(time.max() - time.min())
    idx = below[0]
    return float(time[idx] - time.min())


def _summarize_discharge_cycle(cycle_index: int, cycle: dict) -> dict:
    data = cycle["data"]
    voltage = np.asarray(data["Voltage_measured"], dtype=float)
    current = np.asarray(data["Current_measured"], dtype=float)
    temperature = np.asarray(data["Temperature_measured"], dtype=float)
    time = np.asarray(data["Time"], dtype=float)

    return {
        "cycle_index": cycle_index,
        "ambient_temperature_c": cycle.get("ambient_temperature"),
        "capacity_ah": float(data["Capacity"]),
        "soh": float(data["Capacity"]) / RATED_CAPACITY_AH,
        "discharge_duration_s": float(time.max() - time.min()) if time.size else np.nan,
        "voltage_mean": float(voltage.mean()) if voltage.size else np.nan,
        "voltage_min": float(voltage.min()) if voltage.size else np.nan,
        "voltage_std": float(voltage.std()) if voltage.size else np.nan,
        "voltage_slope": _voltage_slope(time, voltage),
        "time_to_knee_voltage_s": _time_to_voltage_threshold(time, voltage, VOLTAGE_KNEE_THRESHOLD_V),
        "current_mean": float(current.mean()) if current.size else np.nan,
        "current_std": float(current.std()) if current.size else np.nan,
        "temperature_mean": float(temperature.mean()) if temperature.size else np.nan,
        "temperature_max": float(temperature.max()) if temperature.size else np.nan,
    }


def parse_battery_file(mat_path: Path) -> pd.DataFrame:
    battery_id = mat_path.stem  # e.g. "B0005"
    mat = sio.loadmat(mat_path, simplify_cells=True)
    cycles = mat[battery_id]["cycle"]

    rows = []
    discharge_idx = 0
    for cycle in cycles:
        if cycle["type"] != "discharge":
            continue
        row = _summarize_discharge_cycle(discharge_idx, cycle)
        row["battery_id"] = battery_id
        rows.append(row)
        discharge_idx += 1

    df = pd.DataFrame(rows)
    # 열화 진행률(0=신품, 1=완전열화 기준선)을 함께 남겨 EDA에서 바로 쓸 수 있게 함
    df["cycles_seen"] = df["cycle_index"] + 1
    return df


def build_dataset(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    frames = [parse_battery_file(p) for p in sorted(raw_dir.glob("B*.mat"))]
    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    return full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--out", default="data/processed/battery_cycles.csv", type=Path)
    args = parser.parse_args()

    df = build_dataset(args.raw_dir, args.out)
    print(f"{len(df)} discharge cycles parsed from {df['battery_id'].nunique()} batteries")
    print(df.groupby("battery_id")["soh"].agg(["count", "min", "max"]))
    print(f"saved -> {args.out}")
