"""
TUẦN 2 — Sinh dataset Q&A cho fine-tune
=========================================
Input : haui_chunks.jsonl  (120 chunks)
Output: haui_qa_dataset.jsonl  — cặp hỏi-đáp format Alpaca
        haui_qa_train.jsonl    — 80% để train
        haui_qa_val.jsonl      — 20% để validate

Dùng Gemini Flash (miễn phí, ~1500 req/ngày) để sinh Q&A.
Mỗi chunk → 4 cặp Q&A → tổng ~480 cặp.

Cài đặt:
    pip install google-generativeai

Chạy:
    python week2_generate_qa.py
"""

import json
import os
import random
import re
import time
from pathlib import Path

import google.generativeai as genai

# ── Cấu hình ──────────────────────────────────────────────────────────────────
# Thêm nhiều key vào đây, lấy miễn phí tại https://aistudio.google.com/app/apikey
API_KEYS = [
    "AIzaSyA_UOUxBHpIJfC3u1yYDDMS5F8fuwqeMWw",   # key 1
    "AIzaSyB8swhyIcuKN4IBEBN0W5Z2E8-7WIapM68",   # key 2
    "AIzaSyCTroj3MqNshrrXPlXvHnkICc1WMo-S3kg",   # key 3
    "AIzaSyCou5OAbexSL7VGsJ1B6EhbdR40XpE4Ues",   # key 4
    "AIzaSyD6aXcYmKOegaJKB9kF_k1hENvb5Ucc3VE",   # key 5
    "AIzaSyDWhi0EYlQ83yQRB-zwAEiHT9suScMqCm8",  # key 6
    "AIzaSyCTq9KRBZAy6iZDWmM3O8xGkdHyswkRLOs",   # key 7
]

CHUNKS_FILE    = "haui_chunks.jsonl"
OUTPUT_FILE    = "2haui_qa_dataset.jsonl"
TRAIN_FILE     = "2haui_qa_train.jsonl"
VAL_FILE       = "haui_qa_val.jsonl"

QA_PER_CHUNK   = 4     # số cặp Q&A sinh ra mỗi chunk
DELAY_SECONDS  = 6.0   # nghỉ giữa các request (10 req/phút → an toàn với free tier)
TRAIN_RATIO    = 0.8   # 80% train / 20% val

# ── System prompt của chatbot (dùng khi fine-tune) ────────────────────────────
CHATBOT_SYSTEM = (
    "Bạn là trợ lý tư vấn tuyển sinh của Đại học Công nghiệp Hà Nội (HAUI). "
    "Hãy trả lời chính xác, thân thiện và ngắn gọn dựa trên thông tin tuyển sinh của trường."
)

# ── Prompt sinh Q&A ───────────────────────────────────────────────────────────
QA_GENERATION_PROMPT = """Bạn là chuyên gia tạo dữ liệu huấn luyện AI cho chatbot tư vấn tuyển sinh đại học.

Dưới đây là một đoạn thông tin về tuyển sinh của Trường Đại học Công nghiệp Hà Nội (HAUI):

[NỘI DUNG]
{chunk_text}

Hãy tạo ra ĐÚNG {n} cặp câu hỏi - câu trả lời dựa trên nội dung trên.

Yêu cầu:
- Câu hỏi: viết như thí sinh thật đặt câu hỏi (tự nhiên, đa dạng cách hỏi)
- Câu trả lời: chính xác theo nội dung, thân thiện, đầy đủ thông tin
- Đa dạng: hỏi về các khía cạnh khác nhau trong đoạn văn
- KHÔNG bịa thông tin không có trong đoạn văn

Trả về JSON array đúng format sau (chỉ JSON, không giải thích thêm):
[
  {{
    "question": "câu hỏi của thí sinh",
    "answer": "câu trả lời của tư vấn viên HAUI"
  }},
  ...
]"""


# ── Quản lý API Key ───────────────────────────────────────────────────────────
class KeyManager:
    def __init__(self, keys: list):
        self.keys        = [k for k in keys if k and "YOUR_KEY" not in k]
        self.current_idx = 0
        self.dead_keys   = set()  # key đã hết quota ngày/tháng

        if not self.keys:
            raise ValueError(
                "Không có API key hợp lệ!\n"
                "Hãy điền key vào list API_KEYS ở đầu file.\n"
                "Lấy key miễn phí tại: https://aistudio.google.com/app/apikey"
            )
        print(f"✓ Có {len(self.keys)} API key sẵn sàng")

    @property
    def current_key(self) -> str:
        return self.keys[self.current_idx]

    def get_model(self):
        """Trả về Gemini model với key hiện tại."""
        genai.configure(api_key=self.current_key)
        return genai.GenerativeModel("gemini-2.5-flash")

    def rotate(self) -> bool:
        """
        Đánh dấu key hiện tại là dead, chuyển sang key tiếp theo.
        Trả về True nếu còn key khả dụng, False nếu hết sạch.
        """
        self.dead_keys.add(self.current_key)
        key_short = self.current_key[:12] + "..."
        print(f"\n  [KEY MANAGER] Key [{key_short}] hết quota → chuyển key mới")

        # Tìm key tiếp theo chưa dead
        for i in range(len(self.keys)):
            next_idx = (self.current_idx + 1 + i) % len(self.keys)
            if self.keys[next_idx] not in self.dead_keys:
                self.current_idx = next_idx
                new_short = self.keys[next_idx][:12] + "..."
                print(f"  [KEY MANAGER] Đang dùng key [{new_short}] ✓\n")
                return True

        # Hết sạch key
        print("\n  [KEY MANAGER] ⚠️  TẤT CẢ API KEY ĐÃ HẾT QUOTA!")
        print("  Thêm key mới vào list API_KEYS và chạy lại.")
        return False

    def status(self):
        alive = len(self.keys) - len(self.dead_keys)
        print(f"  API Keys: {alive}/{len(self.keys)} còn hoạt động "
              f"| {len(self.dead_keys)} hết quota")


# ── Hàm sinh Q&A từ 1 chunk ───────────────────────────────────────────────────
def generate_qa_for_chunk(key_manager: KeyManager, chunk: dict, n: int = QA_PER_CHUNK) -> list:
    """Gọi Gemini sinh Q&A từ 1 chunk, trả về list dict {question, answer}."""
    prompt = QA_GENERATION_PROMPT.format(
        chunk_text=chunk["text"][:1500],  # giới hạn để tránh vượt token
        n=n,
    )

    MAX_RETRIES = 5
    raw = ""

    for attempt in range(MAX_RETRIES):
        model = key_manager.get_model()  # lấy model với key hiện tại
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                    # KHÔNG dùng response_mime_type — gemini-2.5-flash hay lỗi với option này
                },
            )
            raw = resp.text.strip()

            # Strip markdown fences nếu có: ```json ... ``` hoặc ``` ... ```
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()

            pairs = json.loads(raw)

            # Validate và lọc
            valid = []
            if isinstance(pairs, list):
                for p in pairs:
                    q = str(p.get("question", "")).strip()
                    a = str(p.get("answer", "")).strip()
                    if len(q) > 10 and len(a) > 20:
                        valid.append({"question": q, "answer": a})
            return valid

        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse lỗi: {e} | raw[:200]: {raw[:200]}")

            # Fallback: thử extract các object hoàn chỉnh bằng regex
            try:
                pattern = r'\{\s*"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
                matches = re.findall(pattern, raw, re.DOTALL)
                if matches:
                    print(f"    [RECOVER] Khôi phục được {len(matches)} cặp Q&A")
                    return [{"question": q, "answer": a} for q, a in matches]
            except Exception:
                pass
            return []

        except Exception as e:
            err_str = str(e)

            if "429" in err_str:
                # Phân biệt: hết quota ngày/tháng vs rate limit tạm thời (per-minute)
                is_quota_exhausted = (
                    "quota" in err_str.lower()
                    or "Resource has been exhausted" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                )

                if is_quota_exhausted:
                    # Hết quota ngày/tháng → rotate key ngay, không chờ
                    has_next = key_manager.rotate()
                    if not has_next:
                        return []  # hết sạch key, dừng
                    continue  # retry với key mới

                else:
                    # Rate limit tạm thời (per-minute) → chờ rồi retry
                    wait = 65  # mặc định 65s
                    match = re.search(r"retry in ([\d.]+)s", err_str)
                    if match:
                        wait = int(float(match.group(1))) + 5  # cộng thêm 5s buffer

                    print(f"\n    [RATE LIMIT] Vượt quota phút! Đợi {wait}s... (lần {attempt + 1}/{MAX_RETRIES})")
                    for remaining in range(wait, 0, -10):
                        print(f"    ⏳ Còn {remaining}s...", end="\r")
                        time.sleep(min(10, remaining))
                    print("    ✓ Thử lại!                    ")
                    continue  # retry

            # Lỗi khác → bỏ qua chunk này
            print(f"    [LỖI] {e}")
            return []

    print(f"    [FAIL] Hết {MAX_RETRIES} lần thử, bỏ chunk này")
    return []


# ── Chuyển sang format Alpaca (chuẩn cho fine-tune) ──────────────────────────
def to_alpaca_format(qa: dict, category: str) -> dict:
    """
    Format Alpaca — chuẩn được dùng bởi Unsloth, LLaMA-Factory, Axolotl.

    Kèm theo ShareGPT format để linh hoạt.
    """
    return {
        # ── Alpaca format ──
        "instruction": qa["question"],
        "input": "",
        "output": qa["answer"],
        "system": CHATBOT_SYSTEM,
        "category": category,

        # ── ShareGPT format ──
        "conversations": [
            {"from": "system",    "value": CHATBOT_SYSTEM},
            {"from": "human",     "value": qa["question"]},
            {"from": "assistant", "value": qa["answer"]},
        ],
    }


# ── Lưu file ──────────────────────────────────────────────────────────────────
def save_jsonl(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(data):>4} bản ghi → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("TUẦN 2 — Sinh dataset Q&A cho fine-tune")
    print("=" * 55)

    # Khởi tạo KeyManager
    try:
        key_manager = KeyManager(API_KEYS)
    except ValueError as e:
        print(f"\n[LỖI] {e}")
        return

    print(f"✓ Gemini 2.5 Flash sẵn sàng")

    # Tải chunks
    if not Path(CHUNKS_FILE).exists():
        print(f"[LỖI] Không tìm thấy {CHUNKS_FILE}")
        print("Hãy chạy step1_prepare_chunks.py trước")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f if l.strip()]
    print(f"✓ Tải {len(chunks)} chunks")

    # Thống kê phân bố chunks
    cat_counts = {}
    for c in chunks:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    print("\nPhân bố chunks:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:3d} chunks | {cat}")

    # Ước tính
    estimated = len(chunks) * QA_PER_CHUNK
    est_minutes = len(chunks) * DELAY_SECONDS / 60
    print(f"\nDự kiến sinh  : ~{estimated} cặp Q&A")
    print(f"Thời gian ước tính: ~{est_minutes:.0f} phút (không tính thời gian chờ rate limit)")
    print(f"Delay mỗi request : {DELAY_SECONDS}s")
    print()

    # ── Sinh Q&A ──────────────────────────────────────────────────────────────
    all_qa = []
    errors = 0

    for i, chunk in enumerate(chunks):
        cat = chunk["category"]
        print(f"[{i+1:3d}/{len(chunks)}] [{cat}] chunk_{chunk['chunk_id'][-6:]}...", end=" ")

        pairs = generate_qa_for_chunk(key_manager, chunk, QA_PER_CHUNK)

        if pairs:
            for pair in pairs:
                all_qa.append(to_alpaca_format(pair, cat))
            print(f"→ {len(pairs)} Q&A")
        else:
            errors += 1
            print("→ [bỏ qua]")

        # Lưu checkpoint mỗi 20 chunks (phòng trường hợp bị ngắt giữa chừng)
        if (i + 1) % 20 == 0:
            save_jsonl(all_qa, OUTPUT_FILE + ".checkpoint")
            print(f"  [CHECKPOINT] Đã lưu {len(all_qa)} Q&A")
            key_manager.status()

        time.sleep(DELAY_SECONDS)

    print(f"\n{'='*55}")
    print(f"✓ Sinh xong: {len(all_qa)} cặp Q&A ({errors} chunks lỗi)")

    # ── Shuffle và split train/val ─────────────────────────────────────────────
    random.seed(42)
    random.shuffle(all_qa)
    split      = int(len(all_qa) * TRAIN_RATIO)
    train_data = all_qa[:split]
    val_data   = all_qa[split:]

    print("\nLưu file:")
    save_jsonl(all_qa,     OUTPUT_FILE)
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(val_data,   VAL_FILE)

    # ── Thống kê cuối ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("THỐNG KÊ DATASET:")
    print(f"  Tổng Q&A   : {len(all_qa)}")
    print(f"  Train set  : {len(train_data)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val set    : {len(val_data)} ({(1-TRAIN_RATIO)*100:.0f}%)")

    cat_qa = {}
    for item in all_qa:
        cat_qa[item["category"]] = cat_qa.get(item["category"], 0) + 1
    print("\nQ&A theo category:")
    for cat, cnt in sorted(cat_qa.items(), key=lambda x: -x[1]):
        print(f"  {cnt:4d} cặp | {cat}")

    key_manager.status()

    print(f"\n✓ Dataset sẵn sàng cho fine-tune!")
    print(f"  File train: {TRAIN_FILE}")
    print(f"  File val  : {VAL_FILE}")
    print("\nBước tiếp theo: Tuần 3 — Fine-tune trên Google Colab")


if __name__ == "__main__":
    main()