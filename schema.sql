-- pgvector eklentisini aktif et
CREATE EXTENSION IF NOT EXISTS vector;

-- Yüklenen dosyaları tutan tablo
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

-- Dosyalardan çıkarılan metin parçalarını (chunk) ve embedding vektörlerini tutan tablo
-- NOT: VECTOR(384) boyutu all-MiniLM-L6-v2 modeline göredir.
-- Farklı bir embedding modeli kullanırsanız bu boyutu değiştirmeniz gerekir.
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);

-- Benzerlik aramasını hızlandırmak için index (cosine similarity)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
