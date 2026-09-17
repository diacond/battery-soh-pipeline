"""
외부 데이터셋(CALCE) 홀드아웃 평가.

NASA 배터리 4개로 학습한 최종 모델(soh_model.joblib)을 재학습 없이
학습에 쓰지 않은 CALCE CS2 셀에 적용해 일반화 성능을 측정한다.

주의
  - 정격 용량이 다르다(2.0Ah vs 1.1Ah). SOH를 비율로 정의해 비교는 가능하다.
  - CALCE에는 온도 실측치가 없어 온도 피처 3개를 25°C 가정값으로 채웠다.
    완전히 공정한 비교가 아니라 이 제약 안에서의 근사 실험이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.model.train import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate(model, df: pd.DataFrame) -> dict:
    per_battery = []
    for battery_id, g in df.groupby("battery_id"):
        pred = model.predict(g[FEATURE_COLUMNS])
        per_battery.append(
            {
                "battery_id": battery_id,
                "n_cycles": len(g),
                "mae": mean_absolute_error(g[TARGET_COLUMN], pred),
                "r2": r2_score(g[TARGET_COLUMN], pred),
                "actual_soh_mean": float(g[TARGET_COLUMN].mean()),
                "predicted_soh_mean": float(pred.mean()),
            }
        )

    overall_pred = model.predict(df[FEATURE_COLUMNS])
    overall = {
        "n_cycles": len(df),
        "mae": mean_absolute_error(df[TARGET_COLUMN], overall_pred),
        "r2": r2_score(df[TARGET_COLUMN], overall_pred),
    }
    return {"per_battery": per_battery, "overall": overall}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/processed/soh_model.joblib", type=Path)
    parser.add_argument("--data", default="data/processed/calce_cycles.csv", type=Path)
    parser.add_argument("--out", default="data/processed/external_calce_eval.json", type=Path)
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.data)

    results = evaluate(model, df)

    print("=== CALCE 셀별 결과 (NASA 4개 배터리로 학습한 모델을 그대로 사용) ===")
    for row in results["per_battery"]:
        print(
            f"  {row['battery_id']:<10} n={row['n_cycles']:<5} "
            f"MAE={row['mae']:.4f}  R2={row['r2']:.3f}  "
            f"실제 평균 SOH={row['actual_soh_mean']:.3f}  예측 평균 SOH={row['predicted_soh_mean']:.3f}"
        )
    print(f"\n전체: MAE={results['overall']['mae']:.4f}  R2={results['overall']['r2']:.3f} (n={results['overall']['n_cycles']})")

    print("\n참고: NASA 내부 LOBO 평균은 MAE=0.008, R2=0.982였다. 위 수치와 이 값의 격차가")
    print("곧 '같은 실험실 배터리 간 일반화'와 '완전히 다른 배터리로의 일반화' 사이의 차이다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장 -> {args.out}")
