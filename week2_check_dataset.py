"""
Kiểm tra chất lượng dataset Q&A sau khi sinh
Chạy sau week2_generate_qa.py để đảm bảo data sạch trước khi fine-tune.

Chạy:
    python week2_check_dataset.py
"""

import json
import random
from pathlib import Path

QA_FILE    = "haui_qa_dataset.jsonl"
TRAIN_FILE = "haui_qa_train.jsonl"
VAL_FILE   = "haui_qa_val.jsonl"

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_dataset():
    print("=" * 55)
    print("KIỂM TRA CHẤT LƯỢNG DATASET Q&A")
    print("=" * 55)

    if not Path(QA_FILE).exists():
        print(f"[LỖI] Không tìm thấy {QA_FILE}")
        print("Hãy chạy week2_generate_qa.py trước")
        return

    data  = load_jsonl(QA_FILE)
    train = load_jsonl(TRAIN_FILE)
    val   = load_jsonl(VAL_FILE)

    print(f"\nSố lượng:")
    print(f"  Tổng  : {len(data)}")
    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")

    # ── Kiểm tra độ dài ──────────────────────────────────────────────────────
    q_lens = [len(d["instruction"]) for d in data]
    a_lens = [len(d["output"]) for d in data]

    print(f"\nĐộ dài câu hỏi (ký tự):")
    print(f"  Trung bình : {sum(q_lens)//len(q_lens)}")
    print(f"  Min / Max  : {min(q_lens)} / {max(q_lens)}")

    print(f"\nĐộ dài câu trả lời (ký tự):")
    print(f"  Trung bình : {sum(a_lens)//len(a_lens)}")
    print(f"  Min / Max  : {min(a_lens)} / {max(a_lens)}")

    # ── Cảnh báo câu trả lời quá ngắn (<50 ký tự) ───────────────────────────
    short_answers = [d for d in data if len(d["output"]) < 50]
    if short_answers:
        print(f"\n[CẢNH BÁO] {len(short_answers)} câu trả lời ngắn dưới 50 ký tự:")
        for d in short_answers[:3]:
            print(f"  Q: {d['instruction'][:60]}")
            print(f"  A: {d['output']}")
            print()

    # ── Phân bố theo category ────────────────────────────────────────────────
    cat_counts = {}
    for d in data:
        cat_counts[d["category"]] = cat_counts.get(d["category"], 0) + 1

    print("\nPhân bố Q&A theo category:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 5)
        print(f"  {cnt:4d} | {cat:<30} {bar}")

    # ── Xem ngẫu nhiên 5 mẫu ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("5 MẪU NGẪU NHIÊN:")
    print("=" * 55)
    samples = random.sample(data, min(5, len(data)))
    for i, s in enumerate(samples, 1):
        print(f"\n[{i}] Category: {s['category']}")
        print(f"Q: {s['instruction']}")
        print(f"A: {s['output'][:200]}{'...' if len(s['output']) > 200 else ''}")

    # ── Kiểm tra trùng lặp ───────────────────────────────────────────────────
    questions = [d["instruction"] for d in data]
    unique_q  = set(questions)
    if len(unique_q) < len(questions):
        dupes = len(questions) - len(unique_q)
        print(f"\n[CẢNH BÁO] Có {dupes} câu hỏi bị trùng lặp — nên xóa trước khi fine-tune")
    else:
        print(f"\n✓ Không có câu hỏi trùng lặp")

    # ── Kết luận ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    if len(data) >= 300:
        print("✓ Dataset ĐỦ để fine-tune (>= 300 cặp)")
    elif len(data) >= 150:
        print("⚠ Dataset ở mức TỐI THIỂU (150-299 cặp) — sẽ fine-tune được nhưng chất lượng vừa phải")
    else:
        print("✗ Dataset QUÁ ÍT (<150 cặp) — nên crawl thêm data rồi sinh Q&A lại")

    print(f"\nBước tiếp theo: Upload {TRAIN_FILE} và {VAL_FILE} lên Google Colab để fine-tune")


if __name__ == "__main__":
    check_dataset()
