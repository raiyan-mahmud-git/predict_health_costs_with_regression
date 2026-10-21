import urllib.error

import pandas as pd
import pytest

from health_costs.data import load_dataset, validate_dataset


def test_validation_requires_schema(sample_data):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(sample_data.drop(columns="expenses"))


def test_local_data_load_retains_duplicate_rows(sample_data, tmp_path):
    frame = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
    path = tmp_path / "insurance.csv"
    frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert len(loaded) == len(frame)
    assert loaded.duplicated().sum() == 1


def test_cache_is_used_without_network(sample_data, tmp_path, monkeypatch):
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    sample_data.to_csv(cache_dir / "insurance.csv", index=False)

    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be called for an existing cache")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    assert len(load_dataset(cache_dir=cache_dir)) == len(sample_data)


def test_offline_error_is_actionable(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    with pytest.raises(RuntimeError, match="provide a local CSV"):
        load_dataset(cache_dir=tmp_path / "empty", timeout_seconds=0.1)
