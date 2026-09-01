"""
NASA + CALCE 혼합 학습 실험 — 데이터 누수 없이 "다양한 배터리를 섞어
학습하면 새로운 배터리에 더 잘 통하는가"를 확인한다.

이전 두 실험과의 관계(정직하게 구분해둔다):
  1. `src/model/eval_external.py` — "NASA만으로 학습한 모델이 완전히
     낯선 배터리(CALCE)에 얼마나 통하는가"를 측정한 순수 일반화 테스트.
     이 결과(MAE 0.055, R² 0.693, 편향 +0.053)는 그대로 유지된다 —
     이 실험이 그 결과를 대체하거나 무효화하지 않는다.
  2. `src/model/experiment_normalized_feature.py` — 위 편향의 원인을
     "배터리 스펙 차이에 따른 피처 스케일 불일치"로 보고 정규화 피처로
     고쳐봤지만 가설이 기각됐다.
  3. (이 스크립트) — 그렇다면 아예 "학습 단계에서부터 CALCE 같은 다른
     종류의 배터리를 일부 섞어서 배우면 어떨까"를 확인한다. 이건 위
     1번과는 다른 질문이다 — "NASA만으로 완전한 미지의 배터리에
     통하는가"가 아니라 "다양한 배터리로 학습하면 또 다른 새 배터리에
     더 잘 통하는가"를 묻는다.

**데이터 누수 방지가 핵심이다.** CALCE 4개 셀을 전부 학습에 넣고 같은
CALCE로 다시 평가하면 자기 자신으로 시험 보는 것과 같아 의미가 없다.
그래서 기존 LOBO 방식을 그대로 확장했다: NASA 배터리 4개는 항상 학습에
포함하고, CALCE 셀 4개 중 3개를 추가로 학습에 섞은 뒤, 학습에 쓰인
적 없는 나머지 1개 CALCE 셀로만 평가한다. 이걸 CALCE 4개 각각 돌아가며
4번 반복해 평균을 낸다 — NASA 배터리로 했던 LOBO와 정확히 같은 원칙을
CALCE 쪽에도 적용한 것이다.
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

    # R2가 1.000에 가깝게 나오면(실제로 그렇다) 곧이곧대로 "완벽히 해결됐다"고
    # 믿기 전에, 이게 진짜 일반화인지 CALCE 4개 셀이 서로 너무 닮아서
    # 생긴 착시인지부터 의심해야 한다 - 이 프로젝트 전체를 관통하는 태도다.
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
