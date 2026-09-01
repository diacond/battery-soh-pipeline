"""
배터리 SOH(State of Health) 예측 베이스라인 모델.

핵심 설계 결정 (README/면접에서 설명할 부분):
1. 평가 방식 - Leave-One-Battery-Out(LOBO) cross-validation을 쓴다.
   같은 배터리의 사이클을 무작위로 train/test에 섞으면 인접 사이클끼리
   정보가 새어(leakage) 실제보다 성능이 부풀려진다. 실무에서 이 모델이
   맞닥뜨릴 상황은 "한 번도 본 적 없는 새 배터리의 초기 사이클만 보고
   미래 열화를 예측하는 것"이므로, 배터리 단위로 통째로 홀드아웃한다.
2. RUL(Remaining Useful Life) 정의 - NASA PCoE 벤치마크 관례를 따라
   SOH가 0.7(정격 용량의 70%) 밑으로 떨어지는 시점을 End-of-Life로 본다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

EOL_SOH_THRESHOLD = 0.70
FEATURE_COLUMNS = [
    "cycles_seen",
    "ambient_temperature_c",
    "discharge_duration_s",
    "voltage_mean",
    "voltage_min",
    "voltage_std",
    "voltage_slope",
    "time_to_knee_voltage_s",
    "current_mean",
    "current_std",
    "temperature_mean",
    "temperature_max",
]
TARGET_COLUMN = "soh"


def compute_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """배터리별로 EOL(SOH<0.7 최초 도달) 사이클까지 남은 사이클 수를 라벨링."""
    out = []
    for battery_id, g in df.groupby("battery_id"):
        g = g.sort_values("cycles_seen").reset_index(drop=True)
        below = g.index[g["soh"] < EOL_SOH_THRESHOLD]
        eol_cycle = g.loc[below[0], "cycles_seen"] if len(below) else g["cycles_seen"].max()
        g["rul_cycles"] = (eol_cycle - g["cycles_seen"]).clip(lower=0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def leave_one_battery_out_eval(df: pd.DataFrame) -> list[dict]:
    results = []
    batteries = sorted(df["battery_id"].unique())
    for held_out in batteries:
        train = df[df["battery_id"] != held_out]
        test = df[df["battery_id"] == held_out]

        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        pred = model.predict(test[FEATURE_COLUMNS])

        results.append(
            {
                "held_out_battery": held_out,
                "n_test_cycles": len(test),
                "mae": mean_absolute_error(test[TARGET_COLUMN], pred),
                "r2": r2_score(test[TARGET_COLUMN], pred),
            }
        )
    return results


def train_final_model(df: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    return model


if __name__ == "__main__":
    processed = Path("data/processed/battery_cycles.csv")
    df = pd.read_csv(processed)
    df = compute_rul_labels(df)
    df.to_csv(processed, index=False)  # rul_cycles 컬럼 반영

    cv_results = leave_one_battery_out_eval(df)
    print(json.dumps(cv_results, indent=2, ensure_ascii=False))

    mean_mae = np.mean([r["mae"] for r in cv_results])
    mean_r2 = np.mean([r["r2"] for r in cv_results])
    print(f"\nLOBO 평균 MAE: {mean_mae:.4f} (SOH 스케일), 평균 R2: {mean_r2:.3f}")

    Path("data/processed/cv_results.json").write_text(
        json.dumps(cv_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    import joblib

    final_model = train_final_model(df)
    Path("data/processed").mkdir(exist_ok=True, parents=True)
    joblib.dump(final_model, "data/processed/soh_model.joblib")
    print("최종 모델(전체 데이터 학습) 저장 -> data/processed/soh_model.joblib")
