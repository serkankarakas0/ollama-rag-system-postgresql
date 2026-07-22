import os
import shutil
import tempfile

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from chunking import extract_text, chunk_text
from embeddings import embed_texts
from db import insert_document, insert_chunks, list_documents, delete_document
from rag import run_rag_pipeline

app = FastAPI(title="Lokal RAG Sistemi")

# Lokalde çalıştığı için CORS'u tamamen serbest bırakıyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Dosyayı diske geçici olarak kaydeder, metnini çıkarır, parçalara böler,
    embedding'lerini oluşturur ve PostgreSQL'e kaydeder.
    Dosya kalıcı olarak saklanmaz; sadece işlenip veritabanına yazılır.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya türü. İzin verilenler: {ALLOWED_EXTENSIONS}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        if not chunks:
            raise HTTPException(status_code=400, detail="Dosyadan metin çıkarılamadı.")

        embeddings = embed_texts(chunks)

        document_id = insert_document(file.filename)
        chunks_with_embeddings = [
            (idx, content, embedding)
            for idx, (content, embedding) in enumerate(zip(chunks, embeddings))
        ]
        insert_chunks(document_id, chunks_with_embeddings)

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
        }
    finally:
        os.remove(tmp_path)


@app.get("/documents")
def get_documents():
    return list_documents()


@app.delete("/documents/{document_id}")
def remove_document(document_id: int):
    delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}


@app.get("/ask")
def ask(question: str = Query(...), document_id: int | None = Query(None)):
    """
    SSE (Server-Sent Events) endpoint'i. LLM'in izlediği her adımı
    anlık olarak (streaming) frontend'e gönderir.
    """
    return StreamingResponse(
        run_rag_pipeline(question, document_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Basit frontend'i aynı sunucudan servis etmek için (opsiyonel)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
