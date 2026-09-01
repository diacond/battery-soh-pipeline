from pathlib import Path

import joblib
import pytest

from src.model.eval_external import evaluate
from src.pipeline.parse_calce import build_dataset


@pytest.fixture(scope="module")
def calce_df(tmp_path_factory):
    """106개 엑셀 파일 파싱은 몇 분 걸리므로, 모듈 내 테스트가 한 번만 실행하고 공유한다."""
    raw_dir = Path("data/external/calce")
    out_path = tmp_path_factory.mktemp("calce") / "calce_cycles.csv"
    return build_dataset(raw_dir, out_path)


def test_build_dataset_produces_expected_cells(calce_df):
    assert set(calce_df["battery_id"].unique()) == {"CS2_35", "CS2_36", "CS2_37", "CS2_38"}
    # CALCE는 NASA보다 정격 용량이 작고(1.1Ah) 훨씬 낮은 SOH까지 시험됐다.
    assert calce_df["soh"].between(0, 1.2).all()
    assert len(calce_df) > 1000


def test_nasa_model_generalizes_partially_to_calce(calce_df):
    """완전한 실패도, 완벽한 성공도 아님을 확인하는 회귀 테스트.

    NASA 4개 배터리로 학습한 모델을 재학습 없이 그대로 CALCE에 돌렸을 때,
    같은 실험실 내부 LOBO 검증(R2 0.982)보다는 확실히 나쁘지만 그래도
    우연보다는 나은 수준(R2 > 0.5)이어야 한다. 이 범위를 벗어나면
    (모델이 완전히 무너지거나, 반대로 이상하게 NASA 내부 수준까지
    좋아지면) parse_calce.py의 피처 추출 로직이 깨졌다는 신호로 본다.
    """
    model = joblib.load("data/processed/soh_model.joblib")

    results = evaluate(model, calce_df)

    assert 0.5 < results["overall"]["r2"] < 0.95
    assert results["overall"]["mae"] < 0.15
