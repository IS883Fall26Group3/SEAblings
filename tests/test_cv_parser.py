from pathlib import Path

import pytest

from core.cv_parser import ScannedDocumentError, clean_text, extract_text

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


def make_pdf(with_image: bool) -> bytes:
    """A one-page PDF with no text: a picture of a page if with_image (like a scan), else blank."""
    import io
    from pypdf import PdfWriter
    from pypdf.generic import (DecodedStreamObject, DictionaryObject, NameObject, NumberObject)
    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    draw = DecodedStreamObject()
    draw.set_data(b"q 612 0 0 792 0 0 cm /Im0 Do Q" if with_image else b"")
    page[NameObject("/Contents")] = writer._add_object(draw)
    if with_image:
        img = DecodedStreamObject()
        img.set_data(b"\xff\xff\xff")
        img.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
                    NameObject("/Width"): NumberObject(1), NameObject("/Height"): NumberObject(1),
                    NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                    NameObject("/BitsPerComponent"): NumberObject(8)})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): DictionaryObject(
            {NameObject("/Im0"): writer._add_object(img)})})
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_scanned_pdf_says_so_and_what_to_upload_instead():
    with pytest.raises(ScannedDocumentError, match="job description looks like a scanned PDF") as e:
        extract_text(make_pdf(with_image=True), "jd.pdf", what="job description")
    assert "text-based PDF" in str(e.value) and "DOCX or TXT" in str(e.value)


def test_blank_pdf_without_images_is_not_called_scanned():
    with pytest.raises(ValueError, match="Couldn't read enough text") as e:
        extract_text(make_pdf(with_image=False), "cv.pdf")
    assert not isinstance(e.value, ScannedDocumentError)
