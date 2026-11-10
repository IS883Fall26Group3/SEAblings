"""The English-only check on CVs, job descriptions and roles (core/validation.py)."""
from pathlib import Path

import pytest

from core.cv_parser import extract_text
from core.validation import english_only, language_of

DATA = Path(__file__).resolve().parents[1] / "data" / "sample"

SPANISH = ("Analista de datos con cinco años de experiencia en el sector financiero. Responsable de la "
           "elaboración de informes para la dirección y de los modelos de previsión de ventas con Excel y "
           "SQL. Trabajo en equipo con las áreas de finanzas y operaciones para mejorar los procesos.")
INDONESIAN = ("Analis data dengan pengalaman lima tahun di bidang keuangan. Bertanggung jawab untuk "
              "membuat laporan bulanan dan model perkiraan penjualan dengan Excel dan SQL. Bekerja sama "
              "dengan tim keuangan dan operasional untuk memperbaiki proses yang ada di perusahaan ini.")


@pytest.mark.parametrize("text, language", [
    ("นักวิเคราะห์ข้อมูล มีประสบการณ์ 5 ปี ด้าน SQL และ Excel ในบริษัทพลังงาน", "Thai"),
    ("データアナリスト。SQLとExcelを使った売上予測の経験が5年あります。", "Japanese"),
    ("数据分析师，拥有五年使用SQL和Excel进行销售预测的经验。", "Chinese"),
    ("데이터 분석가, SQL과 Excel을 사용한 5년 경력.", "Korean"),
    (SPANISH, "Spanish"),
    (INDONESIAN, "Indonesian"),
])
def test_other_languages_are_named(text, language):
    assert language_of(text) == language


@pytest.mark.parametrize("path", sorted(DATA.glob("**/*.txt")), ids=lambda p: p.name)
def test_english_sample_cvs_pass(path):
    assert language_of(path.read_text(encoding="utf-8")) is None


def test_a_few_foreign_words_in_an_english_cv_pass():
    text = ("Data analyst with 5 years in energy. Languages: English, Thai (ภาษาไทย), Spanish (español). "
            "Built SQL models for the planning team and led the move to Power BI at the company.")
    assert language_of(text) is None


def test_role_and_short_text():
    assert language_of("Data analyst") is None
    assert language_of("นักวิเคราะห์ข้อมูล") == "Thai"
    assert language_of("123 ???") is None


def test_thai_canva_pdf_is_read_and_called_thai():
    """Layout mode finds no text in this Canva PDF; the fallback must, or it looks like a scan."""
    text = extract_text((DATA / "cvs" / "Sample_TH_CV.pdf").read_bytes(), "Sample_TH_CV.pdf")
    assert "\x00" not in text
    assert language_of(text) == "Thai"


def test_messages():
    assert english_only("CV", "Thai") == ("This CV looks like it's in Thai. The app only works with "
                                          "English, so please upload or paste an English version.")
    assert english_only("role", "Thai").endswith("so please enter the role in English.")
