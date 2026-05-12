"""
BƯỚC 1 — Làm sạch & tạo chunks từ 3 file JSONL thô
=====================================================
Input : 3 file JSONL (haui_rag_raw_data, haui_rag_data_v2, haui_debug_v2)
Output: haui_chunks.jsonl  — mỗi dòng là 1 chunk sẵn sàng embed

Chạy:
    python step1_prepare_chunks.py
"""

import json
import re
from pathlib import Path

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # tokens ≈ ký tự / 2  →  ~1000 ký tự / chunk
CHUNK_OVERLAP = 80    # overlap giữa 2 chunk liên tiếp (tránh mất ngữ cảnh)

INPUT_FILES = [
    "haui_rag_raw_data.jsonl",
    "haui_rag_data_v2.jsonl",
    "haui_debug_v2.jsonl",
    "haui_week1_data.jsonl",    # ← thêm sau khi chạy week1_crawl_more.py
]
OUTPUT_FILE = "haui_chunks.jsonl"


# ── Hàm tiện ích ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Loại bỏ rác đầu trang (ngày tháng, số view...) và chuẩn hoá khoảng trắng."""
    lines = text.strip().splitlines()
    clean = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # Bỏ dòng chỉ là ngày/giờ  vd "08/01/2026 07:00:00"
        if re.fullmatch(r'\d{2}/\d{2}/\d{4}.*', ln):
            continue
        # Bỏ dòng chỉ là số đơn lẻ (view count, id...)
        if re.fullmatch(r'\d{1,6}', ln):
            continue
        clean.append(ln)
    return "\n".join(clean)


def table_to_prose(table: list[dict]) -> str:
    """
    Chuyển 1 bảng (list of dict) → văn xuôi.
    Bảng 72 ngành → 72 câu mô tả, dễ retrieve hơn nhiều so với JSON.
    """
    lines = []
    for row in table:
        parts = []
        for k, v in row.items():
            k, v = str(k).strip(), str(v).strip()
            if k and v and v not in ("nan", "NaN", ""):
                parts.append(f"{k}: {v}")
        if parts:
            lines.append(", ".join(parts) + ".")
    return "\n".join(lines)


def split_into_chunks(text: str, source_meta: dict,
                      size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Cắt text thành các chunk theo đơn vị 'từ' (word-level).
    Mỗi chunk kèm metadata để truy vết nguồn sau này.
    """
    words = text.split()
    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end        = min(start + size, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "chunk_id"  : f"{source_meta['doc_id']}_c{idx:03d}",
            "doc_id"    : source_meta["doc_id"],
            "url"       : source_meta["url"],
            "title"     : source_meta["title"],
            "category"  : source_meta["category"],
            "chunk_idx" : idx,
            "text"      : chunk_text,
            "word_count": len(words[start:end]),
        })
        idx   += 1
        start += size - overlap  # trượt cửa sổ, giữ overlap

    return chunks


# ── Logic chính ───────────────────────────────────────────────────────────────

def load_and_merge(filenames: list[str]) -> list[dict]:
    """
    Đọc tất cả file, loại bỏ bản ghi trùng URL,
    ưu tiên giữ bản ghi có nhiều ký tự nhất (chứa bảng).
    """
    seen: dict[str, dict] = {}   # url → record tốt nhất

    for fname in filenames:
        path = Path(fname)
        if not path.exists():
            print(f"  [BỎ QUA] Không tìm thấy: {fname}")
            continue

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                url  = rec.get("url", "")
                text = rec.get("text_content", "")

                # Lấy bảng từ bất kỳ key nào có thể tồn tại
                tables = (rec.get("tables_data")
                          or rec.get("tables_raw")
                          or [])

                # Gom bảng thành văn xuôi và nối vào text
                prose_parts = []
                for tbl in tables:
                    p = table_to_prose(tbl)
                    if p:
                        prose_parts.append(p)
                if prose_parts:
                    text = text + "\n\n[Dữ liệu bảng]\n" + "\n\n".join(prose_parts)

                rec["_merged_text"] = text

                # Giữ bản ghi dài nhất cho mỗi URL
                if url not in seen or len(text) > len(seen[url]["_merged_text"]):
                    seen[url] = rec

    records = list(seen.values())
    print(f"  → {len(records)} bản ghi duy nhất sau khi merge")
    return records


def build_chunks(records: list[dict]) -> list[dict]:
    all_chunks = []

    for i, rec in enumerate(records):
        raw_text = rec.get("_merged_text", rec.get("text_content", ""))
        text     = clean_text(raw_text)

        if not text.strip():
            print(f"  [CẢNH BÁO] Record {i+1} rỗng sau khi làm sạch, bỏ qua.")
            continue

        meta = {
            "doc_id"  : f"doc_{i:03d}",
            "url"     : rec.get("url", ""),
            "title"   : rec.get("title", ""),
            "category": rec.get("category", ""),
        }

        chunks = split_into_chunks(text, meta)
        all_chunks.extend(chunks)
        print(f"  doc_{i:03d} [{meta['category']}]: "
              f"{len(text):,} ký tự → {len(chunks)} chunks")

    return all_chunks


def save_chunks(chunks: list[dict], output: str = OUTPUT_FILE):
    with open(output, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\n✓ Đã lưu {len(chunks)} chunks → {output}")

    # Thống kê nhanh
    word_counts = [c["word_count"] for c in chunks]
    print(f"  Trung bình: {sum(word_counts)//len(word_counts)} từ/chunk")
    print(f"  Min: {min(word_counts)} | Max: {max(word_counts)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("BƯỚC 1 — Chuẩn bị chunks")
    print("=" * 55)

    print("\n[1/3] Đọc & merge 3 file JSONL...")
    records = load_and_merge(INPUT_FILES)

    print("\n[2/3] Làm sạch text & cắt chunks...")
    chunks = build_chunks(records)

    print("\n[3/3] Lưu file...")
    save_chunks(chunks)

    print("\nBước 1 hoàn thành. Chạy tiếp: python step2_embed_index.py")