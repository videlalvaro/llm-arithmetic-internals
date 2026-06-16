from __future__ import annotations

from pathlib import Path

from scripts.goalB3_paper_package_audit import build_package


def test_goalB3_paper_package_current_artifacts_pass_minimum_bar() -> None:
    paths = {
        "broad_cross_seed": Path("docs/goalB3_benchmark_cross_seed_summary.json"),
        "gcd_cross_seed": Path("docs/goalB3_gcd_benchmark_cross_seed_summary.json"),
        "deepmind_cross_seed": Path("docs/goalB3_deepmind_gcd_div_lcm_cross_seed_summary.json"),
        "deepmind_provenance": Path("docs/goalB3_deepmind_gcd_div_lcm_provenance_audit.json"),
        "deepmind_source_audit": Path("docs/goalB3_deepmind_source_audit.json"),
        "adversarial": Path("docs/goalB3_operand_repair.json"),
        "qwen_transfer": Path("docs/goalB3_qwen_strict_transfer.json"),
        "causal_cross_seed": Path("docs/goalB3_causal_interchange_cross_seed.json"),
        "gcd_causal_cross_seed": Path("docs/goalB3_causal_interchange_gcd_cross_seed.json"),
    }

    package = build_package(paths)

    assert package["verdict"] == "NEURIPS_PACKAGE_MINIMUM_BAR_PASS_WITH_CAVEATS"
    assert all(package["gates"].values())
    assert "gcd" in package["causal_interchange"]["ops"]
    assert package["deepmind_cross_seed"]["n_locked_total"] >= 1000
    assert package["deepmind_cross_seed"]["op_checks"]["gcd"]["min_lift"] >= 0.20
    assert package["deepmind_provenance"]["n_negative_fires"] == 0
