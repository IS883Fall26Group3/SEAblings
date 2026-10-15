from pathlib import Path

import pytest

from core.cv_parser import extract_text

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
