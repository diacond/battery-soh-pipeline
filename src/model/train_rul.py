"""
RUL(Remaining Useful Life) 예측 모델.

정비 현장에서는 "지금 위험한가"보다 "몇 사이클 더 쓸 수 있는가"가 중요하다.
한 모델로 SOH와 RUL을 함께 맞추기보다 목적별로 모델을 나눠 단순하게 유지했다.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.model.train import FEATURE_COLUMNS, compute_rul_labels

RUL_TARGET_COLUMN = "rul_cycles"


def leave_one_battery_out_eval_rul(df: pd.DataFrame) -> list[dict]:
    results = []
    for held_out in sorted(df["battery_id"].unique()):
        train = df[df["battery_id"] != held_out]
        test = df[df["battery_id"] == held_out]

        model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
        model.fit(train[FEATURE_COLUMNS], train[RUL_TARGET_COLUMN])
        pred = model.predict(test[FEATURE_COLUMNS])

        results.append(
            {
                "held_out_battery": held_out,
                "n_test_cycles": len(test),
                "mae_cycles": mean_absolute_error(test[RUL_TARGET_COLUMN], pred),
                "r2": r2_score(test[RUL_TARGET_COLUMN], pred),
            }
        )
    return results


if __name__ == "__main__":
    processed = Path("data/processed/battery_cycles.csv")
    df = pd.read_csv(processed)
    if RUL_TARGET_COLUMN not in df.columns:
        df = compute_rul_labels(df)
        df.to_csv(processed, index=False)

    cv_results = leave_one_battery_out_eval_rul(df)
    print(json.dumps(cv_results, indent=2, ensure_ascii=False))

    mean_mae = np.mean([r["mae_cycles"] for r in cv_results])
    mean_r2 = np.mean([r["r2"] for r in cv_results])
    print(f"\nLOBO 평균 MAE: {mean_mae:.1f} cycles, 평균 R2: {mean_r2:.3f}")

    Path("data/processed/cv_results_rul.json").write_text(
        json.dumps(cv_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    final_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    final_model.fit(df[FEATURE_COLUMNS], df[RUL_TARGET_COLUMN])
    joblib.dump(final_model, "data/processed/rul_model.joblib")
    print("RUL 모델(전체 데이터 학습) 저장 -> data/processed/rul_model.joblib")
