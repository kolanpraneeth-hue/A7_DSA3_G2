"""
Text extraction and preprocessing utilities.
Supports TXT, PDF and DOCX files.
"""

import re
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

# Ensure required NLTK data is available
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


def extract_text(file_path: str) -> str:
    """Extract plain text from .txt, .pdf or .docx."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        parts = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(parts)

    if suffix in (".docx", ".doc"):
        from docx import Document

        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .pdf or .docx")


def clean_text(text: str) -> str:
    """Basic cleaning: normalise whitespace and remove control characters."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences and drop very short fragments."""
    text = clean_text(text)
    raw = sent_tokenize(text)
    sentences = []
    for s in raw:
        s = s.strip()
        # Keep sentences with at least a few words
        if len(s.split()) >= 4:
            sentences.append(s)
    return sentences


def load_and_chunk(file_path: str) -> list[str]:
    """Full pipeline: extract → clean → sentence chunks."""
    text = extract_text(file_path)
    return split_sentences(text)
