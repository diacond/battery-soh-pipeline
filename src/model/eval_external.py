"""
외부 데이터셋(CALCE) 홀드아웃 평가 — "진짜 새 배터리" 테스트.

train.py의 LOBO 검증과 train_final_model()이 만드는 최종 모델은 전부
NASA PCoE의 배터리 4개 안에서만 이뤄진다는 한계가 있었다(README/
PROJECT_OVERVIEW.md 참고). 이 스크립트는 그 최종 모델(soh_model.joblib)을
재학습하지 않고 그대로 가져와서, 학습에 단 한 번도 쓰이지 않은 완전히
다른 제조사·실험실의 배터리(CALCE CS2 시리즈)에 그대로 predict()를
돌려본다. 새 모델을 만드는 게 아니라, "이미 만든 모델이 진짜 미지의
배터리 앞에서 얼마나 버티는지"를 확인하는 게 목적이다.

정직하게 미리 밝혀둘 것: CALCE 데이터는 NASA 데이터와 두 가지가
근본적으로 다르다(자세한 설명은 parse_calce.py 상단 참고).
  - 정격 용량이 다르다 (2.0Ah vs 1.1Ah) - SOH 비율로 정규화했으니
    이 자체는 큰 문제가 아니다.
  - 온도 실측치가 없다 - ambient_temperature_c/temperature_mean/
    temperature_max 세 피처를 전부 가정값(25도 고정)으로 채워 넣었다.
    이건 모델 입장에서 "한 번도 본 적 없는 값 조합"이 아니라 "정보량이
    없는 상수"를 준 것에 가깝다. 즉 이 실험은 "완벽하게 공정한 재현"이
    아니라 "이런 제약 안에서 그나마 최대한 맞춰본 근사 실험"이라는 걸
    결과를 읽을 때 감안해야 한다.
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
