"""
BƯỚC 2 — Embed chunks & lưu vào ChromaDB
==========================================
Input : haui_chunks.jsonl  (output của bước 1)
Output: ./chroma_db/        (thư mục vector database)

Embedding model: BAAI/bge-m3
  - Hỗ trợ tiếng Việt rất tốt
  - Miễn phí, chạy local, không cần API key
  - ~570MB, tự động tải về lần đầu

Chạy:
    pip install chromadb sentence-transformers
    python step2_embed_index.py
"""

import json
from pathlib import Path

# ── Import ────────────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Thiếu thư viện. Chạy:")
    print("  pip install chromadb sentence-transformers")
    raise

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CHUNKS_FILE    = "haui_chunks.jsonl"
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "haui_admission"
EMBED_MODEL    = "BAAI/bge-m3"
BATCH_SIZE     = 32   # số chunk embed cùng lúc (giảm nếu hết RAM)


# ── Hàm tiện ích ──────────────────────────────────────────────────────────────

def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Đã tải {len(chunks)} chunks từ {path}")
    return chunks


def get_collection(chroma_dir: str, collection_name: str):
    """Khởi tạo ChromaDB persistent client."""
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    # Xoá collection cũ nếu có (để reindex sạch)
    try:
        client.delete_collection(collection_name)
        print(f"  Đã xoá collection cũ '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


def embed_and_index(chunks: list[dict], collection, model: SentenceTransformer):
    """Embed theo batch và upsert vào ChromaDB."""
    total   = len(chunks)
    indexed = 0

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids   = [c["chunk_id"] for c in batch]

        # Metadata phải là dict[str, str|int|float|bool]
        metas = [
            {
                "doc_id"   : c["doc_id"],
                "url"      : c["url"],
                "title"    : c["title"][:200],      # giới hạn độ dài
                "category" : c["category"],
                "chunk_idx": c["chunk_idx"],
            }
            for c in batch
        ]

        # Embed (bge-m3 tự động normalize)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metas,
        )

        indexed += len(batch)
        print(f"  [{indexed}/{total}] đã index...", end="\r")

    print(f"\n  ✓ Index hoàn thành: {indexed} chunks")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("BƯỚC 2 — Embedding & Index vào ChromaDB")
    print("=" * 55)

    print(f"\n[1/4] Tải embedding model '{EMBED_MODEL}'...")
    print("      (Lần đầu ~570MB, sẽ cache cho lần sau)")
    model = SentenceTransformer(EMBED_MODEL)
    print("      ✓ Model sẵn sàng")

    print(f"\n[2/4] Tải chunks từ {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)

    print(f"\n[3/4] Khởi tạo ChromaDB tại '{CHROMA_DIR}'...")
    collection = get_collection(CHROMA_DIR, COLLECTION)

    print(f"\n[4/4] Embed & index (batch_size={BATCH_SIZE})...")
    embed_and_index(chunks, collection, model)

    print(f"\n✓ Vector DB sẵn sàng tại '{CHROMA_DIR}/'")
    print(f"  Collection: '{COLLECTION}'")
    print(f"  Tổng vectors: {collection.count()}")
    print("\nBước 2 hoàn thành. Chạy tiếp: python step3_rag_chatbot.py")
