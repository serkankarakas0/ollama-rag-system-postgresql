import os
from pypdf import PdfReader
import docx


def extract_text(file_path: str) -> str:
    """Dosya uzantısına göre metni çıkarır. Desteklenen: .pdf, .docx, .txt, .md"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError(f"Desteklenmeyen dosya türü: {ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Metni karakter bazlı, overlap'li parçalara böler.
    Basit ama etkili bir yöntemdir; kelime sınırında kesmeye çalışır.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Kelime ortasında kesmemek için son boşluğa kadar geri git
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap if end - overlap > start else end

    return chunks
