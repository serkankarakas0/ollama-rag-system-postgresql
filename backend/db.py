import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "rag_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}


def get_connection():
    """Yeni bir PostgreSQL bağlantısı döner."""
    return psycopg2.connect(**DB_CONFIG)


def insert_document(filename: str) -> int:
    """Yeni bir doküman kaydı oluşturur ve id'sini döner."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename) VALUES (%s) RETURNING id;",
                (filename,),
            )
            doc_id = cur.fetchone()[0]
        conn.commit()
    return doc_id


def insert_chunks(document_id: int, chunks_with_embeddings):
    """
    chunks_with_embeddings: [(chunk_index, content, embedding_list), ...]
    Birden fazla chunk'ı tek seferde ekler (toplu insert).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            values = [
                (document_id, idx, content, embedding)
                for idx, content, embedding in chunks_with_embeddings
            ]
            execute_values(
                cur,
                """
                INSERT INTO chunks (document_id, chunk_index, content, embedding)
                VALUES %s
                """,
                values,
                template="(%s, %s, %s, %s::vector)",
            )
        conn.commit()


def list_documents():
    """Yüklenmiş tüm dokümanları listeler."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT d.id, d.filename, d.uploaded_at, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.uploaded_at DESC;
                """
            )
            return cur.fetchall()


def delete_document(document_id: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
        conn.commit()


def search_similar_chunks(query_embedding, document_id: int | None, top_k: int = 4):
    """
    Embedding'e en yakın (cosine distance) chunk'ları döner.
    document_id verilirse sadece o dokümana ait chunk'larda arama yapılır.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if document_id:
                cur.execute(
                    """
                    SELECT c.id, c.content, c.chunk_index, d.filename,
                           1 - (c.embedding <=> %s::vector) AS similarity
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.document_id = %s
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, document_id, query_embedding, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT c.id, c.content, c.chunk_index, d.filename,
                           1 - (c.embedding <=> %s::vector) AS similarity
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, top_k),
                )
            return cur.fetchall()
