import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals.config import load_metals_config
from src.metals.recorder import MetalsRecorder


def _config(tmp_path: Path):
    config = load_metals_config("configs/metals.example.toml")
    return replace(
        config,
        storage=replace(config.storage, root=str(tmp_path / "metals")),
        kalshi=replace(
            config.kalshi,
            credentials_config=str(tmp_path / "missing-credentials.toml"),
            api_key_id_env="TEST_MISSING_KALSHI_KEY_ID",
            private_key_path_env="TEST_MISSING_KALSHI_KEY_PATH",
        ),
    )


def _manifest(tmp_path: Path, run_id: str):
    path = (
        tmp_path
        / "metals"
        / "manifests"
        / "run={}".format(run_id)
        / "manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_manual_recorder_finalize_is_idempotent_and_records_outputs(tmp_path):
    recorder = MetalsRecorder(_config(tmp_path), require_credentials=False)
    recorder.quality.record("raw_records", count=3)

    recorder._finalize("completed", counts={"markets": 2})
    recorder._finalize("failed", error=RuntimeError("must be ignored"))

    manifest = _manifest(tmp_path, recorder.run_id)
    assert manifest["status"] == "completed"
    assert manifest["counts"]["raw_records"] == 3
    assert manifest["counts"]["markets"] == 2
    assert manifest["errors"] == []
    assert any(path.endswith("report.json") for path in manifest["output_paths"])


@pytest.mark.asyncio
async def test_missing_credentials_fail_before_network_and_finalize_manifest(tmp_path):
    recorder = MetalsRecorder(_config(tmp_path), require_credentials=False)

    with pytest.raises(RuntimeError, match="valid Kalshi API key"):
        await recorder.run(duration_seconds=0.01)

    manifest = _manifest(tmp_path, recorder.run_id)
    assert manifest["status"] == "failed"
    assert manifest["errors"]
    assert "valid Kalshi API key" in manifest["errors"][0]["message"]
