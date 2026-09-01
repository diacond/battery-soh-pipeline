from pathlib import Path

import pandas as pd

from src.model.experiment_mixed_training import leave_one_calce_cell_out_with_nasa


def test_mixing_calce_into_training_nearly_solves_same_batch_generalization():
    """CALCE 3개+NASA로 학습해서 CALCE 나머지 1개를 맞히는 건 eval_external.py의
    '완전 미지의 배터리' 테스트보다 훨씬 쉬운 문제라는 걸 확인하는 회귀 테스트.

    같은 제조 배치의 셀 몇 개만 섞어 학습해도 거의 완벽하게(R2>0.99) 맞히는
    반면, eval_external.py의 순수 NASA 학습 모델은 R2 0.693에 그쳤다 - 이
    격차 자체가 "완전히 낯선 배터리 종류 vs 이미 일부를 알고 있는 배터리
    종류"의 난이도 차이를 보여준다.
    """
    nasa = pd.read_csv(Path("data/processed/battery_cycles.csv"))
    calce = pd.read_csv(Path("data/processed/calce_cycles.csv"))

    results = leave_one_calce_cell_out_with_nasa(nasa, calce)

    assert {r["held_out_cell"] for r in results} == {"CS2_35", "CS2_36", "CS2_37", "CS2_38"}
    mean_r2 = sum(r["r2"] for r in results) / len(results)
    mean_mae = sum(r["mae"] for r in results) / len(results)
    assert mean_r2 > 0.99
    assert mean_mae < 0.01
