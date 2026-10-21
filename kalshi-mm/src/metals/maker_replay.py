"""Verify and replay recorded maker inputs without credentials or network."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from dataclasses import fields, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

from .baseline_fair import load_baseline_vol_model
from .config import MetalsConfig
from .maker_policy import MakerEngine, MakerSettings, tiny_maker_config
from .shadow_session import ShadowSession


def verified_inputs(path: Path) -> Iterator[dict[str, Any]]:
    previous = None
    sequence = -1
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            digest = row.pop("record_sha256", None)
            canonical = json.dumps(row, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, default=str)
            if digest != hashlib.sha256(canonical.encode()).hexdigest():
                raise ValueError(f"maker input hash mismatch: {path}:{number}")
            if row.get("previous_record_sha256") != previous:
                raise ValueError(f"maker input chain mismatch: {path}:{number}")
            if type(row.get("ingestion_sequence")) is not int or row["ingestion_sequence"] <= sequence:
                raise ValueError(f"maker input sequence regressed: {path}:{number}")
            previous, sequence = digest, row["ingestion_sequence"]
            yield row


def settings_from_dict(value: dict[str, Any]) -> MakerSettings:
    defaults = MakerSettings()
    known = {field.name for field in fields(MakerSettings)}
    if set(value) != known:
        raise ValueError("maker replay requires the complete frozen settings")
    return MakerSettings(**{key: Decimal(str(item)) if isinstance(getattr(defaults, key), Decimal)
                            else item for key, item in value.items()})


def replay_report(report_path: Path, output_root: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_path = Path(report["baseline_model_path"])
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != report["baseline_model_sha256"]:
        raise ValueError("maker baseline model hash changed")
    model, study_id = load_baseline_vol_model(model_path)
    settings = settings_from_dict(report["settings"])
    paths = [Path(value) for value in report["maker_input_paths"]]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("maker input paths missing or duplicated")
    if any(not path.is_file() for path in paths):
        raise ValueError("maker input segment missing")
    config = tiny_maker_config(MetalsConfig())
    config = replace(config, storage=replace(config.storage, root=str(output_root)))
    sessions = {arm: ShadowSession(config, manifest_context={
        "maker_replay_source": str(report_path.resolve()), "arm": arm,
        "baseline_model_sha256": model_hash, "settings": settings.public_dict(),
    }) for arm in ("manual", "baseline")}
    engine = MakerEngine(model, study_id, settings, sessions)
    count, previous_sequence, stopped = 0, -1, False
    status, failure = "completed", None
    try:
        streams = [verified_inputs(path) for path in paths]
        for event in heapq.merge(*streams, key=lambda row: row["ingestion_sequence"]):
            count += 1
            for key in ("ingestion_sequence", "receive_ts_ns", "local_monotonic_ts_ns"):
                if type(event.get(key)) is not int or event[key] < 0:
                    raise ValueError(f"invalid replay event {key}")
            if event["ingestion_sequence"] <= previous_sequence:
                raise ValueError("non-increasing replay input sequence")
            previous_sequence = event["ingestion_sequence"]
            if event["kind"] == "stop":
                if stopped:
                    raise ValueError("multiple maker stop events")
                stopped = True
                engine.close(event)
            else:
                engine.handle(event)
        if count != report["maker_input_counts"]:
            raise ValueError("maker replay input count differs from capture")
        summary = engine.summary()
        expected = report["summary"]["decision_digest"]
        if summary["decision_digest"] != expected:
            raise ValueError("maker replay decision digest differs from capture")
        # Digest verifies decisions; full comparison also covers lifecycle, fills,
        # unresolved exposure, and marks which are assembled in summary.
        canonical = lambda value: json.dumps(value, sort_keys=True, default=str)
        if canonical(summary) != canonical(report["summary"]):
            raise ValueError("maker replay final state differs from capture")
        result = {"replay_verified": True, "input_records": count,
                  "source_capture_status": report.get("status", "unknown"),
                  "source_error_types": report.get("error_types", []),
                  "trial_research": report.get("trial_research"),
                  "execution_qualified": False,
                  "source_report": str(report_path.resolve()), "summary": summary,
                  "shadow_run_ids": {arm: session.run_id for arm, session in sessions.items()}}
        path = output_root / "maker_replay" / f"run={sessions['manual'].run_id}" / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        return {"replay_verified": True, "input_records": count, "report": str(path),
                "source_capture_status": result["source_capture_status"],
                "execution_qualified": False}
    except BaseException as error:
        status, failure = "failed", error
        raise
    finally:
        for session in sessions.values():
            session.close(status=status, error=failure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(replay_report(args.report, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
