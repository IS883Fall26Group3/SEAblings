from pathlib import Path

import pytest

from core.cv_parser import clean_text, extract_text

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_cv.txt"


def test_reads_txt():
    text = extract_text(SAMPLE.read_bytes(), "cv.txt")
    assert "Financial modeling" in text


def test_rejects_unsupported():
    with pytest.raises(ValueError):
        extract_text(b"x" * 500, "cv.png")


def test_rejects_near_empty():
    with pytest.raises(ValueError):
        extract_text(b"too short", "cv.txt")


def test_clean_text_accepts_pasted_cv():
    pasted = "Jane Doe\n\n\n\nAnalyst   with SQL.  " + "Experience in reporting. " * 10
    text = clean_text(pasted)
    assert "\n\n\n" not in text and "   " not in text


def test_clean_text_rejects_short_paste():
    with pytest.raises(ValueError, match="Paste the full text"):
        clean_text("Just my name")
