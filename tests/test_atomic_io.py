import os

import pandas as pd

from src.utils.atomic_io import atomic_write_csv


def test_atomic_write_csv_writes_correct_content(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    target = tmp_path / "out.csv"
    atomic_write_csv(df, str(target), index=False)

    assert target.exists()
    result = pd.read_csv(target)
    pd.testing.assert_frame_equal(result, df)


def test_atomic_write_csv_leaves_no_temp_files(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    target = tmp_path / "out.csv"
    atomic_write_csv(df, str(target), index=False)

    leftover = [f for f in os.listdir(tmp_path) if f.startswith(".tmp_atomic_")]
    assert leftover == []


def test_atomic_write_csv_overwrites_existing_file(tmp_path):
    target = tmp_path / "out.csv"
    atomic_write_csv(pd.DataFrame({"a": [1]}), str(target), index=False)
    atomic_write_csv(pd.DataFrame({"a": [9, 9, 9]}), str(target), index=False)

    result = pd.read_csv(target)
    assert list(result["a"]) == [9, 9, 9]


def test_atomic_write_csv_creates_missing_directory(tmp_path):
    target = tmp_path / "nested" / "dir" / "out.csv"
    atomic_write_csv(pd.DataFrame({"a": [1]}), str(target), index=False)
    assert target.exists()
