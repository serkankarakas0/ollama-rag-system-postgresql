# Lokal RAG Sistemi (PostgreSQL + pgvector + Ollama)

Bu proje, dosyaları arayüzden yükleyip, PostgreSQL (pgvector) içinde vektör
olarak saklayan ve sorularınıza **tamamen lokal çalışan Ollama** üzerinden
cevap üretirken **her adımı canlı olarak** ekranda gösteren basit bir RAG
sistemidir. İnternet bağlantısı olmadan, hiçbir API anahtarı gerekmeden
çalışır — embedding modeli de LLM de bilgisayarınızda çalışır.

## Mimari

```
[Tarayıcı: index.html]
      │  (1) Dosya yükle (multipart/form-data)
      ▼
[FastAPI backend]
      │  metni çıkar → parçala (chunk) → embed et (lokal, ücretsiz)
      ▼
[PostgreSQL + pgvector]  (chunk + embedding saklanır)

      │  (2) Soru sor (SSE / EventSource)
      ▼
[FastAPI backend]
      │  soruyu embed et → pgvector'da benzerlik araması →
      │  bulunan parçalarla prompt oluştur → Ollama'ya gönder (lokal)
      ▼
[Ollama - lokal LLM]  → cevap → SSE ile anlık olarak frontend'e akıtılır
```

Her adım (`"Soru alındı"`, `"Embedding oluşturuluyor"`, `"Veritabanında
aranıyor"`, `"Ollama'ya gönderiliyor"` vb.) Server-Sent Events (SSE) ile
eşzamanlı olarak arayüze yansır; cevabın kendisi de token token akar.

## 1. Gereksinimler

- Python 3.10+
- PostgreSQL 14+ (lokalde kurulu olmalı)
- [pgvector](https://github.com/pgvector/pgvector) eklentisi
- [Ollama](https://ollama.com/download) (tamamen lokal LLM çalıştırma uygulaması, ücretsiz)

## 2. PostgreSQL + pgvector Kurulumu

### Ubuntu/Debian
```bash
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-16-pgvector   # PostgreSQL sürümünüze göre değişebilir
```

### macOS (Homebrew)
```bash
brew install postgresql pgvector
```

### Veritabanını oluşturun
```bash
psql -U serkankarakas -d postgres -c "CREATE DATABASE rag_db_ollama;"
psql -U serkankarakas -d rag_db_ollama -f schema.sql
```

(Kendi PostgreSQL kullanıcı adınız `serkankarakas` değilse onunla değiştirin.)

`schema.sql` dosyası `CREATE EXTENSION vector;` komutunu da içerdiği için
pgvector otomatik olarak aktif edilir.

## 3. Ollama Kurulumu (Lokal LLM)

1. https://ollama.com/download adresinden macOS uygulamasını indirip kurun.
2. Ollama uygulamasını bir kere açın (arka planda bir servis olarak çalışmaya başlar,
   menü çubuğunda bir simge görürsünüz).
3. Terminalde modeli indirin (M1/8GB RAM için önerilen, ~2GB):
   ```bash
   ollama pull llama3.2:3b
   ```
4. Test etmek isterseniz: `ollama run llama3.2:3b` yazıp bir soru sorabilirsiniz
   (çıkmak için `/bye`).

## 4. Backend Kurulumu

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# .env dosyasını açıp DB bilgilerinizi girin (Ollama için varsayılan ayarlar genelde yeterlidir)
```

## 5. Çalıştırma

Ollama'nın arka planda çalıştığından emin olun (menü çubuğunda simgesi görünmeli),
sonra:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Sonra tarayıcıda şu adresi açın:

```
http://localhost:8000
```

(Frontend backend tarafından otomatik olarak servis edilir; ayrı bir sunucuya gerek yok.)

## 6. Kullanım

1. **Dosya Yükle** panelinden `.pdf`, `.docx`, `.txt` veya `.md` dosyanızı seçip **Yükle**'ye basın.
   Dosya diske kalıcı olarak kaydedilmez; sadece metni çıkarılıp parçalanır,
   embedding'leri oluşturulur ve veritabanına yazılır.
2. **Soru Sor** kutusuna sorunuzu yazın (isterseniz belirli bir dosyayı seçebilirsiniz).
3. **Sor**'a bastığınızda, LLM'in izlediği adımlar ("embedding oluşturuluyor",
   "veritabanında aranıyor", "bulunan parçalar", "Ollama'ya gönderiliyor" vb.)
   canlı olarak 3. panelde akar, cevap da 4. panelde anlık olarak yazılır.

## Notlar / Özelleştirme

- **Embedding modeli**: Varsayılan olarak `all-MiniLM-L6-v2` (384 boyut,
  tamamen lokal ve ücretsiz, ilk çalıştırmada otomatik indirilir) kullanılır.
  Farklı bir model kullanmak isterseniz `.env` içindeki `EMBEDDING_MODEL`
  değerini ve `schema.sql` içindeki `VECTOR(384)` boyutunu değiştirmeniz gerekir.
- **Ollama modeli**: `.env` içindeki `OLLAMA_MODEL` ile değiştirilebilir. Farklı
  bir model denemek isterseniz önce `ollama pull <model-adı>` ile indirin, sonra
  `.env`'i güncelleyin. 8GB RAM'de 3B civarı modeller (örn. `qwen2.5:3b`) en
  güvenli seçimdir; 7B+ modeller yavaş çalışabilir ya da bilgisayarı zorlayabilir.
- **Chunk boyutu**: `.env` içindeki `CHUNK_SIZE` / `CHUNK_OVERLAP` ile ayarlanır.
- **Birden fazla dosya**: Aynı anda birden fazla dosya yükleyebilir, arayüzden
  belirli bir dosya üzerinde arama yapabilir ya da tüm dosyalarda arattırabilirsiniz.

## Dosya Yapısı

```
rag-system/
├── schema.sql              # PostgreSQL tabloları + pgvector extension
├── backend/
│   ├── main.py              # FastAPI endpoint'leri (/upload, /documents, /ask)
│   ├── db.py                 # PostgreSQL bağlantı ve sorgular
│   ├── chunking.py           # PDF/DOCX/TXT'den metin çıkarma + parçalama
│   ├── embeddings.py         # Lokal embedding modeli (sentence-transformers)
│   ├── rag.py                # RAG pipeline + adım adım SSE streaming + Ollama çağrısı
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html            # Dosya yükleme + canlı adım gösterimi arayüzü
```
