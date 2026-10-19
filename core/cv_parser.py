"""Turn an uploaded CV or job description (PDF, DOCX, TXT) into plain text. Runs fully in memory."""
import io
import re

from docx import Document
from pypdf import PdfReader

SUPPORTED = (".pdf", ".docx", ".txt")


def extract_text(file_bytes: bytes, filename: str, max_chars: int = 15_000,
                 what: str = "CV") -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
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
    if len(text) < 200:
        raise ValueError(f"Couldn't read enough text from this {what}. {hint}")
    return text[:max_chars]
