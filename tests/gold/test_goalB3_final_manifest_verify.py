from __future__ import annotations

from pathlib import Path

from scripts.goalB3_final_manifest_verify import verify


def test_goalB3_final_manifest_matches_current_defaults_and_artifacts() -> None:
    payload = verify(Path("docs/goalB3_final_frozen_manifest.json"))

    assert payload["verdict"] == "FINAL_MANIFEST_VERIFY_PASS"
    assert payload["errors"] == []
    assert "gcd=0.20" in payload["benchmark_defaults"]["pair_threshold"]
    assert "gcd=0.20" in payload["causal_defaults"]["pair_threshold"]
