"""
NASA + CALCE 혼합 학습 실험.

질문: 여러 종류의 배터리를 섞어 학습하면 새 배터리를 더 잘 맞히는가?
eval_external.py(NASA만으로 학습한 모델의 외부 일반화)와는 다른 질문이며,
그 결과를 대체하지 않는다.

데이터 누수 방지: NASA 4개는 항상 학습에 넣고, CALCE 4개 중 3개를 추가로
학습한 뒤 남은 1개로만 평가한다. 이를 CALCE 셀마다 반복한다
(leave-one-cell-out).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.model.train import FEATURE_COLUMNS, TARGET_COLUMN


def leave_one_calce_cell_out_with_nasa(nasa_df: pd.DataFrame, calce_df: pd.DataFrame) -> list[dict]:
    results = []
    cells = sorted(calce_df["battery_id"].unique())
    for held_out in cells:
        train = pd.concat([nasa_df, calce_df[calce_df["battery_id"] != held_out]], ignore_index=True)
        test = calce_df[calce_df["battery_id"] == held_out]

        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        pred = model.predict(test[FEATURE_COLUMNS])

        results.append(
            {
                "held_out_cell": held_out,
                "n_train_calce_cells": calce_df["battery_id"].nunique() - 1,
                "n_test_cycles": len(test),
                "mae": mean_absolute_error(test[TARGET_COLUMN], pred),
                "r2": r2_score(test[TARGET_COLUMN], pred),
                "bias": float((pred - test[TARGET_COLUMN]).mean()),
            }
        )
    return results


if __name__ == "__main__":
    nasa = pd.read_csv("data/processed/battery_cycles.csv")
    calce = pd.read_csv("data/processed/calce_cycles.csv")

    results = leave_one_calce_cell_out_with_nasa(nasa, calce)

    print("=== NASA 4개 + CALCE 3개로 학습 -> 남은 CALCE 1개로 평가 (4라운드) ===")
    for r in results:
        print(
            f"  held-out={r['held_out_cell']:<10} n={r['n_test_cycles']:<5} "
            f"MAE={r['mae']:.4f}  R2={r['r2']:.3f}  편향={r['bias']:+.4f}"
        )

    mean_mae = sum(r["mae"] for r in results) / len(results)
    mean_r2 = sum(r["r2"] for r in results) / len(results)
    mean_bias = sum(r["bias"] for r in results) / len(results)
    print(f"\n평균: MAE={mean_mae:.4f}  R2={mean_r2:.3f}  편향={mean_bias:+.4f}")

    print("\n비교 대상 (NASA만으로 학습해서 CALCE 4개 전부에 그대로 돌린 결과):")
    print("  MAE=0.0550  R2=0.693  편향=+0.0534")

    verdict = (
        "혼합 학습이 도움됐다(편향/오차가 줄었다)"
        if mean_mae < 0.0550 and abs(mean_bias) < abs(0.0534)
        else "혼합 학습만으로는 뚜렷한 개선이 없었다"
    )
    print(f"\n결론: {verdict}")

    # R²가 1.000에 가까우면, 실제 일반화인지 CALCE 셀끼리 너무 닮아서 생긴
    # 착시인지 확인한다.
    print("\n=== 참고: 이 결과가 왜 이렇게 좋은지 의심해보기 ===")
    pivot = calce.pivot_table(index="cycles_seen", columns="battery_id", values="soh").dropna()
    calce_cross_cell_std = pivot.std(axis=1).mean()
    npivot = nasa.pivot_table(index="cycles_seen", columns="battery_id", values="soh").dropna()
    nasa_cross_cell_std = npivot.std(axis=1).mean()
    print(f"같은 사이클 수에서 셀 간 SOH 표준편차 - CALCE: {calce_cross_cell_std:.4f}, NASA: {nasa_cross_cell_std:.4f}")
    print(
        "CALCE 4개 셀은 같은 제조 배치·같은 실험실·같은 프로토콜로 시험돼 서로 매우 닮아 있다.\n"
        "그래서 이 실험이 보여주는 건 '전혀 다른 화학 조성의 배터리에도 통한다'가 아니라,\n"
        "'같은 종류(제조 배치)의 배터리 몇 개만 섞어 학습해도 그 종류의 나머지 배터리는\n"
        "거의 완벽하게 맞힐 수 있다'는 것에 가깝다 - eval_external.py의 '완전 미지의 배터리'\n"
        "실험과는 답하는 질문 자체가 다르다는 걸 결과를 읽을 때 감안해야 한다."
    )

    Path("data/processed/experiment_mixed_training.json").write_text(
        json.dumps(
            {
                "per_fold": results,
                "mean": {"mae": mean_mae, "r2": mean_r2, "bias": mean_bias},
                "baseline_nasa_only": {"mae": 0.0550, "r2": 0.693, "bias": 0.0534},
                "verdict": verdict,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n저장 -> data/processed/experiment_mixed_training.json")
