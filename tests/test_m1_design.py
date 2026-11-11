"""Tests for the M1 design engine - the only milestone that runs on a CPU."""

from datetime import date

from betacard.m1_design import (build_catalog_plan, build_personalisation,
                                contrast_ratio, detect_script, ordinal,
                                parse_dob, plan_as_records)

TODAY = date(2026, 11, 12)


def person(name="Aayush", dob="24/08/2008"):
    return build_personalisation(name, dob, today=TODAY)


def test_ordinals():
    expected = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 11: "11th",
                12: "12th", 13: "13th", 21: "21st", 24: "24th", 101: "101st"}
    for n, word in expected.items():
        assert ordinal(n) == word


def test_script_detection():
    assert detect_script("Aayush") == "latin"
    assert detect_script("\u092a\u094d\u0930\u093f\u092f\u093e") == "devanagari"
    assert detect_script("\u041c\u0430\u0440\u0438\u044f") == "cyrillic"


def test_dob_parsing_accepts_a_real_date():
    assert parse_dob("24/08/2008") is not None


def test_dob_parsing_rejects_junk():
    import pytest
    from betacard.m1_design import InputError
    for junk in ("", "   ", "not-a-date", "31/02/2008"):
        with pytest.raises(InputError):
            parse_dob(junk)


def test_contrast_is_computed_correctly():
    assert contrast_ratio("#FFFFFF", "#000000") > 20
    assert contrast_ratio("#FFFFFF", "#FFFFFF") == 1.0


def test_catalog_has_twelve_cards():
    plan = build_catalog_plan("sess_abc123", person())
    assert len(plan) == 12


def test_catalog_is_deterministic():
    a = build_catalog_plan("sess_abc123", person())
    b = build_catalog_plan("sess_abc123", person())
    assert [c["design_id"] for c in a] == [c["design_id"] for c in b]
    assert [c["message"]["text"] for c in a] == [c["message"]["text"] for c in b]


def test_catalog_differs_across_sessions():
    a = build_catalog_plan("sess_one", person())
    b = build_catalog_plan("sess_two", person())
    assert [c["design_id"] for c in a] != [c["design_id"] for c in b]


def test_catalog_diversity():
    plan = build_catalog_plan("sess_abc123", person())
    triples = {(c["design"].palette.palette_id,
                c["design"].layout.layout_id,
                c["design"].motif.motif_id) for c in plan}
    assert len(triples) == 12, "every card must be visually distinct"
    assert len({c["design_id"] for c in plan}) == 12


def test_every_card_has_a_message_with_the_name():
    plan = build_catalog_plan("sess_abc123", person())
    for card in plan:
        assert "Aayush" in card["message"]["text"]


def test_plan_as_records_is_json_safe():
    import json
    records = plan_as_records(build_catalog_plan("sess_abc123", person()))
    json.dumps(records)
