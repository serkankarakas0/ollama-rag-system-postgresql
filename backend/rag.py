import os
import json
import ollama

from embeddings import embed_query
from db import search_similar_chunks

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))

ollama_client = ollama.Client(host=OLLAMA_HOST)


def sse_event(event_type: str, data: dict) -> str:
    """Server-Sent Events formatında satır üretir."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Kaynak: {c['filename']} - Parça {c['chunk_index']}]\n{c['content']}"
        for c in chunks
    )
    return (
        "Aşağıda verilen BAĞLAM'a dayanarak soruyu cevapla. "
        "Sadece BAĞLAM'da yazan bilgiyi kullan; kelimeleri, sayıları ve ifadeleri "
        "olduğu gibi aktar, yorum katma, çıkarım yapma veya eksik ayrıntı uydurma. "
        "Eğer cevap bağlamda yoksa, bilmediğini açıkça belirt, uydurma bilgi verme.\n\n"
        f"BAĞLAM:\n{context}\n\n"
        f"SORU: {question}\n\n"
        "CEVAP:"
    )


def run_rag_pipeline(question: str, document_id: int | None = None):
    """
    Generator: her adımı SSE olarak yield eder, sonunda cevabı token token akıtır.
    Frontend bu adımları eşzamanlı olarak ekranda gösterir.
    LLM olarak tamamen lokal çalışan Ollama kullanılır (internet gerekmez).
    """
    try:
        yield sse_event("step", {"message": f'Soru alındı: "{question}"'})

        yield sse_event("step", {"message": "Soru için embedding oluşturuluyor..."})
        query_embedding = embed_query(question)

        yield sse_event(
            "step",
            {"message": "PostgreSQL (pgvector) üzerinde benzerlik araması yapılıyor..."},
        )
        chunks = search_similar_chunks(query_embedding, document_id, top_k=TOP_K)

        if not chunks:
            yield sse_event(
                "step",
                {"message": "Uyarı: Hiç ilgili parça bulunamadı. Doküman yüklendi mi?"},
            )
        else:
            preview = [
                {
                    "kaynak": c["filename"],
                    "parça_no": c["chunk_index"],
                    "benzerlik": round(float(c["similarity"]), 3),
                    "önizleme": c["content"][:120] + "...",
                }
                for c in chunks
            ]
            yield sse_event(
                "step",
                {
                    "message": f"{len(chunks)} ilgili parça bulundu.",
                    "details": preview,
                },
            )

        yield sse_event("step", {"message": "Prompt (bağlam + soru) oluşturuluyor..."})
        prompt = build_prompt(question, chunks)

        yield sse_event(
            "step",
            {"message": f"Ollama'ya istek gönderiliyor (lokal model: {OLLAMA_MODEL})..."},
        )

        stream = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.2},
        )

        yield sse_event("step", {"message": "Cevap üretiliyor (lokal olarak)..."})

        full_answer = ""
        for chunk in stream:
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                full_answer += delta
                yield sse_event("answer_chunk", {"text": delta})

        yield sse_event("done", {"full_answer": full_answer})

    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            yield sse_event(
                "error",
                {
                    "message": (
                        f"'{OLLAMA_MODEL}' modeli bulunamadı. Terminalde şunu çalıştırıp "
                        f"modeli indirin: ollama pull {OLLAMA_MODEL}"
                    )
                },
            )
        else:
            yield sse_event("error", {"message": str(e)})
    except Exception as e:
        yield sse_event(
            "error",
            {
                "message": (
                    f"Ollama'ya bağlanılamadı ({OLLAMA_HOST}). Ollama uygulamasının/servisinin "
                    f"çalıştığından emin olun. Detay: {e}"
                )
            },
        )
