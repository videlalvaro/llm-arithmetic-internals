from __future__ import annotations

from scripts.goalB3_provenance_audit import audit_records


def test_goalB3_provenance_audit_passes_required_fired_fields() -> None:
    records = [
        {
            "example_id": "target_ok",
            "fired": True,
            "is_target": 1,
            "op_source": "activation",
            "operand_source": "activation",
            "answer_source": "python_from_decoded_tuple",
        },
        {
            "example_id": "negative_abstain",
            "fired": False,
            "is_target": 0,
        },
    ]

    summary = audit_records(records, [])

    assert summary["verdict"] == "PROVENANCE_AUDIT_PASS"
    assert summary["n_fired"] == 1
    assert summary["n_negative_fires"] == 0
    assert summary["n_bad_fired_provenance"] == 0


def test_goalB3_provenance_audit_fails_parser_provenance() -> None:
    records = [
        {
            "example_id": "bad",
            "fired": True,
            "is_target": 1,
            "op_source": "text_parser",
            "operand_source": "activation",
            "answer_source": "python_from_decoded_tuple",
        }
    ]

    summary = audit_records(records, [])

    assert summary["verdict"] == "PROVENANCE_AUDIT_FAIL"
    assert summary["n_bad_fired_provenance"] == 1
