"""Turn an uploaded CV or job description (PDF, DOCX, TXT) into plain text. Runs fully in memory."""
import io
import re

from docx import Document
from pypdf import PdfReader

SUPPORTED = (".pdf", ".docx", ".txt")
MIN_CHARS = 200


class ScannedDocumentError(ValueError):
    """A PDF that is pictures of pages, with no text layer to read."""


def _page_text(page) -> str:
    # Layout mode keeps each printed line on its own line; the default can run a whole page
    # into a few long lines (some macOS PDFs), which hides headings and bullets.
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:  # an unusual page (no content stream, odd fonts): treat as no text
        return ""


def _has_images(reader: PdfReader) -> bool:
    """True if any page draws an image. Reads only the page resources, no image data."""
    try:
        for page in reader.pages:
            resources = page.get("/Resources")
            xobjects = resources.get_object().get("/XObject") if resources else None
            for obj in (xobjects.get_object().values() if xobjects else []):
                if obj.get_object().get("/Subtype") == "/Image":
                    return True
    except Exception:
        pass
    return False


def extract_text(file_bytes: bytes, filename: str, max_chars: int = 15_000,
                 what: str = "CV") -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(_page_text(page) for page in reader.pages)
        if len(re.sub(r"\s+", "", text)) < MIN_CHARS and _has_images(reader):
            raise ScannedDocumentError(
                f"This {what} looks like a scanned PDF (a picture of the page), so the app can't read its "
                "text. Upload a text-based PDF (one where you can select the text), a DOCX or TXT file, "
                "or paste the text instead.")
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type. Upload one of: {', '.join(SUPPORTED)}")
    return clean_text(text, max_chars, what, hint="If it's a scanned PDF, upload a text-based "
                                                  "PDF or DOCX instead.")


def clean_text(text: str, max_chars: int = 15_000, what: str = "CV",
               hint: str = "Paste the full text.") -> str:
    """Tidy whitespace and check there's enough text to analyze (used for files and pasted text)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < MIN_CHARS:
        raise ValueError(f"Couldn't read enough text from this {what}. {hint}")
    return text[:max_chars]
