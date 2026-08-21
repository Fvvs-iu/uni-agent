from __future__ import annotations

import json

import pytest

from uni_agent.tasks.deepeyes.preprocess import prepare_deepeyes, validation_positions


def test_validation_positions_are_unique_and_sorted():
    pytest.importorskip("numpy")

    positions = validation_positions(row_count=10, validation_size=3)

    assert positions == sorted(set(positions))
    assert len(positions) == 3
    assert all(0 <= position < 10 for position in positions)


def test_prepare_deepeyes_splits_local_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source_path = tmp_path / "source.parquet"
    rows = [
        {
            "prompt": [{"role": "user", "content": f"<image>Question {index}"}],
            "images": [{"bytes": bytes([index])}],
            "reward_model": {"ground_truth": str(index)},
            "extra_info": {"index": index, "split": "source"},
        }
        for index in range(8)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source_path)

    output_dir = tmp_path / "prepared"
    manifest = prepare_deepeyes(
        output_dir=output_dir,
        source_file=source_path,
        validation_size=2,
        batch_size=3,
    )

    train = pq.read_table(output_dir / "train.parquet").to_pylist()
    validation = pq.read_table(output_dir / "val.parquet").to_pylist()
    saved_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert len(train) == 6
    assert len(validation) == 2
    assert {row["extra_info"]["split"] for row in train} == {"train"}
    assert {row["extra_info"]["split"] for row in validation} == {"validation"}
    assert [row["extra_info"]["index"] for row in validation] == manifest["validation_indices"]
    assert saved_manifest == manifest


def test_prepare_deepeyes_validates_source_schema(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source_path = tmp_path / "invalid.parquet"
    pq.write_table(pa.table({"prompt": ["question"]}), source_path)

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_deepeyes(
            output_dir=tmp_path / "prepared",
            source_file=source_path,
            validation_size=1,
        )
