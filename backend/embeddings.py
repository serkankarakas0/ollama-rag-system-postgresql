import os
from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    """Modeli tek sefer yükleyip tekrar kullanır (lazy singleton)."""
    global _model
    if _model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Birden fazla metni embedding'e çevirir."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Tek bir sorguyu embedding'e çevirir."""
    return embed_texts([text])[0]
