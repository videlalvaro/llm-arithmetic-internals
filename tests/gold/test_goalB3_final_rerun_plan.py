from __future__ import annotations

from pathlib import Path

from scripts.goalB3_final_rerun_plan import build_plan, read_json


def test_goalB3_final_rerun_plan_uses_manifest_frozen_deepmind_tier() -> None:
    manifest = read_json(Path("docs/goalB3_final_frozen_manifest.json"))
    plan = build_plan(manifest, dm_dir="/tmp/deepmind/interpolate", output_prefix="goalB3_final_test")

    names = {item["name"] for item in plan["commands"]}
    assert "deepmind_interpolate_recognized_seed911" in names
    assert "deepmind_interpolate_recognized_seed921" in names
    assert "deepmind_interpolate_recognized_seed931" in names
    assert "deepmind_provenance" in names
    assert "deepmind_interpolate_recognized_replay_provenance_full" in names

    seed911 = next(item for item in plan["commands"] if item["name"] == "deepmind_interpolate_recognized_seed911")
    command = seed911["command"]
    assert "--ops gcd div_remainder lcm" in command
    assert "--pair_threshold mul=0.05" in command
    assert "--pair_threshold gcd=0.20" in command
    assert "--dm_dir /tmp/deepmind/interpolate" in command
    assert "goalB3_final_test_deepmind_interpolate_recognized_seed911.json" in command
    assert "--out_replay_bundles" in command
    assert "goalB3_final_test_deepmind_interpolate_recognized_seed911_replay_bundles.jsonl" in command

    replay = next(item for item in plan["commands"] if item["name"] == "deepmind_interpolate_recognized_replay_provenance_full")
    assert "scripts/goalB3_replay_provenance_audit.py" in replay["command"]
    assert "--require_full" in replay["command"]


def test_goalB3_final_rerun_plan_does_not_commit_local_deepmind_path() -> None:
    manifest = read_json(Path("docs/goalB3_final_frozen_manifest.json"))
    plan = build_plan(manifest, dm_dir=None, output_prefix="goalB3_final_test")

    text = "\n".join(item["command"] for item in plan["commands"])
    assert "$EXTERNAL_VOLUME" not in text
    assert "--dm_dir" not in text
