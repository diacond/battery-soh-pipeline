"""
CALCE(Center for Advanced Life Cycle Engineering, 메릴랜드대) CS2 시리즈
배터리 데이터 파서 — "진짜 새 배터리" 홀드아웃 검증용.

목적: NASA PCoE 4개 배터리로 학습한 최종 SOH 모델(soh_model.joblib)이
한 번도 본 적 없는, 완전히 다른 제조사/실험실/프로토콜의 배터리에도
통하는지 확인하기 위한 외부 데이터셋. src/model/train.py의 LOBO 검증은
"같은 실험 조건 내에서 배터리 개체 간 일반화"만 확인했다는 한계가 있었고,
이 스크립트는 그 한계를 실제로 메우는 시도다.

원본 출처: CALCE Battery Group, University of Maryland. CS2 시리즈
(Panasonic 18650, LiCoO2, 정격 용량 1.1Ah) 셀 4개(CS2_35/36/37/38)의
반복 충/방전 열화 기록. 원본은 Arbin 배터리 테스터가 뽑은 엑셀 파일
묶음이고(셀 1개당 세션(날짜)별로 파일이 나뉘어 있고, 각 파일 안에서
Cycle_Index가 1부터 다시 시작한다), 여기서는 이 원본을 그대로 재배포하는
공개 GitHub 미러(XiuzeZhou/CALCE)에서 받는다.

NASA 데이터와의 호환성 관련 중요한 주의사항(정직하게 남겨둠):
  1. CALCE 원본에는 온도 측정치가 아예 없다(Arbin 로그에 온도 채널이
     없음). NASA 모델은 temperature_mean/temperature_max를 피처로
     쓰기 때문에, 이 스크립트는 CALCE 셀이 상온 항온조에서 시험됐다는
     문헌상의 프로토콜 설명에 근거해 ASSUMED_AMBIENT_TEMP_C를 두 피처에
     그대로 대입한다. 즉 이 두 피처는 "실측"이 아니라 "가정값"이고,
     사이클마다 실제로는 방전 중 발열로 온도가 오르내렸을 텐데 그 변화가
     전혀 반영되지 않는다 — NASA 학습 데이터의 온도 피처는 사이클마다
     30~42도로 실제 변동이 있었던 것과 대조적이다. 이 자체가 두 데이터셋
     사이의 메울 수 없는 차이이자, 외부 홀드아웃 결과를 해석할 때
     감안해야 할 한계다.
  2. 정격 용량이 다르다 (NASA 2.0Ah vs CALCE CS2 1.1Ah). SOH는 두
     데이터셋 모두 "정격 대비 실측 용량 비율"이라는 같은 정의를 쓰므로
     0~1 스케일에서는 비교 가능하지만, 절대 방전 전류·방전 시간 등은
     스케일 자체가 다르다.
  3. 전압 임계값(VOLTAGE_KNEE_THRESHOLD_V=3.0V)은 NASA 코드에서
     그대로 재사용한다 — CS2도 같은 리튬이온(LiCoO2/흑연) 화학 조성이라
     방전 말기 전압 강하 구간이 비슷한 전압대에서 나타나기 때문에 큰
     무리는 아니라고 판단했다.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.parse_mat import _time_to_voltage_threshold, _voltage_slope, VOLTAGE_KNEE_THRESHOLD_V

CALCE_RATED_CAPACITY_AH = 1.1  # CALCE CS2 시리즈 공식 정격 용량 (NASA의 2.0Ah와 다름)
ASSUMED_AMBIENT_TEMP_C = 25.0  # 실측 아님 - CALCE 상온 시험 프로토콜에 근거한 가정값 (한계, 위 docstring 참고)
MIN_DISCHARGE_SAMPLES = 5  # 이보다 적은 샘플만 있는 사이클(캘리브레이션 등)은 노이즈로 보고 제외

_FILENAME_RE = re.compile(r"CS2_\d+_(\d+)_(\d+)_(\d+)\.xlsx$")


def _file_sort_key(path: Path) -> tuple[int, int, int]:
    """파일명의 month_day_2digitYear를 (year, month, day) 정렬키로 변환."""
    m = _FILENAME_RE.search(path.name)
    if not m:
        return (9999, 0, 0)
    month, day, yy = (int(x) for x in m.groups())
    year = 2000 + yy
    return (year, month, day)


def _find_channel_sheet(xls: pd.ExcelFile) -> str | None:
    candidates = [s for s in xls.sheet_names if s.lower().startswith("channel")]
    return candidates[0] if candidates else None


def _summarize_discharge_cycle(cycle_df: pd.DataFrame, global_cycle_index: int, battery_id: str) -> dict | None:
    discharge = cycle_df[cycle_df["Current(A)"] < -0.01]
    if len(discharge) < MIN_DISCHARGE_SAMPLES:
        return None

    time = discharge["Test_Time(s)"].to_numpy(dtype=float)
    voltage = discharge["Voltage(V)"].to_numpy(dtype=float)
    current = discharge["Current(A)"].to_numpy(dtype=float)
    # Discharge_Capacity(Ah)는 파일 전체에 걸친 누적값이라 사이클 경계에서 0으로
    # 리셋되지 않는다. 이번 방전 구간 안에서의 증가폭(max-min)이 실제로 이번
    # 사이클에 방전된 용량이다.
    dis_capacity = discharge["Discharge_Capacity(Ah)"].to_numpy(dtype=float)
    capacity_ah = float(dis_capacity.max() - dis_capacity.min())

    return {
        "battery_id": battery_id,
        "cycles_seen": global_cycle_index,
        "ambient_temperature_c": ASSUMED_AMBIENT_TEMP_C,
        "capacity_ah": capacity_ah,
        "soh": capacity_ah / CALCE_RATED_CAPACITY_AH,
        "discharge_duration_s": float(time.max() - time.min()) if time.size else np.nan,
        "voltage_mean": float(voltage.mean()),
        "voltage_min": float(voltage.min()),
        "voltage_std": float(voltage.std()),
        "voltage_slope": _voltage_slope(time - time.min(), voltage),
        "time_to_knee_voltage_s": _time_to_voltage_threshold(time - time.min(), voltage, VOLTAGE_KNEE_THRESHOLD_V),
        "current_mean": float(current.mean()),
        "current_std": float(current.std()),
        "temperature_mean": ASSUMED_AMBIENT_TEMP_C,
        "temperature_max": ASSUMED_AMBIENT_TEMP_C,
    }


def parse_cell(cell_dir: Path) -> pd.DataFrame:
    battery_id = cell_dir.name  # e.g. "CS2_35"
    files = sorted(cell_dir.glob("*.xlsx"), key=_file_sort_key)

    rows: list[dict] = []
    global_cycle_index = 0
    for path in files:
        xls = pd.ExcelFile(path)
        sheet = _find_channel_sheet(xls)
        if sheet is None:
            continue
        df = pd.read_excel(xls, sheet_name=sheet)
        for _, cycle_df in df.groupby("Cycle_Index", sort=True):
            global_cycle_index += 1
            row = _summarize_discharge_cycle(cycle_df, global_cycle_index, battery_id)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def build_dataset(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    cell_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    frames = [parse_cell(d) for d in cell_dirs]
    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    return full


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/external/calce", type=Path)
    parser.add_argument("--out", default="data/processed/calce_cycles.csv", type=Path)
    args = parser.parse_args()

    df = build_dataset(args.raw_dir, args.out)
    print(f"{len(df)} discharge cycles parsed from {df['battery_id'].nunique()} CALCE cells")
    print(df.groupby("battery_id")["soh"].agg(["count", "min", "max"]))
    print(f"saved -> {args.out}")
