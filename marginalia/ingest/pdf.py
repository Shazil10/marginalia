"""Robust PDF text extraction with multiple fallbacks.

Tries fast text-layer extractors first (PyPDF2, pdfplumber, PyMuPDF), then OCR
as a last resort for scanned papers. Returns up to ``max_chars`` characters,
which is plenty to cover the abstract + methodology where strategy logic lives.
"""

from __future__ import annotations

import os

MIN_TEXT_LEN = 50


def _pypdf2(filepath, max_chars):
    import PyPDF2
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
            if len(text) >= max_chars:
                break
    return text[:max_chars].strip()


def _pdfplumber(filepath, max_chars):
    import pdfplumber
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
            if len(text) >= max_chars:
                break
    return text[:max_chars].strip()


def _pymupdf(filepath, max_chars):
    import fitz  # PyMuPDF
    text = ""
    doc = fitz.open(filepath)
    try:
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
    finally:
        doc.close()
    return text[:max_chars].strip()


def _ocr_pymupdf(filepath, max_chars):
    try:
        import io
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    text = ""
    try:
        doc = fitz.open(filepath)
        try:
            matrix = fitz.Matrix(2, 2)
            for i in range(min(20, len(doc))):
                pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img) + "\n"
                if len(text) >= max_chars:
                    break
        finally:
            doc.close()
    except Exception:
        return ""
    return text[:max_chars].strip()


_EXTRACTORS = [
    ("PyPDF2", _pypdf2),
    ("pdfplumber", _pdfplumber),
    ("PyMuPDF", _pymupdf),
    ("OCR", _ocr_pymupdf),
]


def extract_text_from_pdf(filepath: str, max_chars: int = 8000) -> str:
    """Return up to ``max_chars`` of text from a PDF, trying multiple backends."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    for name, fn in _EXTRACTORS:
        try:
            text = fn(filepath, max_chars)
        except Exception:
            continue
        if text and len(text.strip()) >= MIN_TEXT_LEN:
            return text
    return ""
