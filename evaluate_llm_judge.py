"""
=======================================================================
  HAUI Chatbot Evaluation — LLM-as-a-Judge
=======================================================================
  Đánh giá chatbot RAG fine-tuned theo phương pháp LLM-as-a-Judge:
  - "Quan tòa" 1: Google Gemini 2.0 Flash  (Free)
  - "Quan tòa" 2: Groq  Llama-3.3-70B       (Free)
  So sánh: Baseline (Qwen3-8B gốc qua Ollama) vs HAUI Bot (fine-tuned)

  Cài thư viện trước khi chạy:
      pip install google-generativeai groq requests openpyxl pandas tqdm
=======================================================================
"""

import json
import time
import re
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from google import genai
from groq import Groq

# ──────────────────────────────────────────────
#  ⚙️  CẤU HÌNH  —  ĐIỀN API KEY VÀO ĐÂY
# ──────────────────────────────────────────────
# ─── GEMINI: Danh sách API key (tự đổi khi hết quota) ───────────────
GEMINI_API_KEYS = [
    "YOUR_GEMINI_API_KEY",   # key thứ 1
    "YOUR_GEMINI_API_KEY",   # key thứ 2
    "YOUR_GEMINI_API_KEY",
       # thêm tiếp nếu có...
]
GEMINI_MODEL    = "gemini-3.1-flash-lite-preview"   # model ổn định nhất, free

# ─── GROQ: Danh sách API key (tự đổi khi hết quota) ───────────────────
GROQ_API_KEYS = [
    "YOUR_GROQ_API_KEY",   # key thứ 1
    "YOUR_GROQ_API_KEY",   # key thứ 2 (tạo tại console.groq.com)
    "YOUR_GROQ_API_KEY",   # key thứ 3
]
GROQ_MODEL    = "llama-3.3-70b-versatile"

# ─── CẤU HÌNH CHẠY ──────────────────────────────────────────────
NUM_SAMPLES     = 50                   # số mẫu muốn đánh giá
VAL_DATA_PATH   = "haui_qa_val.jsonl"
_ts             = datetime.now().strftime("%Y%m%d_%H%M")   # timestamp tự động
OUTPUT_EXCEL    = f"ket_qua_danh_gia_{_ts}.xlsx"           # tên file luôn mới → không bị Permission denied
OUTPUT_CSV      = f"ket_qua_danh_gia_{_ts}.csv"
HAUI_BOT_MODEL  = "haui_bot:latest"     # FT2 — model đã fine-tune
BASELINE_MODEL  = "qwen3:8b"            # FT1 — model gốc chưa fine-tune
OLLAMA_URL      = "http://localhost:11434/api/generate"


# ──────────────────────────────────────────────
#  KHỞI TẠO CLIENT
# ──────────────────────────────────────────────
_gemini_key_idx = 0
_gemini_client  = genai.Client(api_key=GEMINI_API_KEYS[0])
_groq_key_idx   = 0
_groq_exhausted = False   # cờ: True = hết sạch key, không thử nữa
groq_client     = Groq(api_key=GROQ_API_KEYS[0])


def _rotate_gemini_key() -> bool:
    """Chuyển sang Gemini key tiếp theo."""
    global _gemini_key_idx, _gemini_client
    _gemini_key_idx += 1
    if _gemini_key_idx >= len(GEMINI_API_KEYS):
        print("  [Gemini] ⚠️ Đã dùng hết tất cả API key! Chuyển sang Groq.")
        return False
    _gemini_client = genai.Client(api_key=GEMINI_API_KEYS[_gemini_key_idx])
    print(f"  [Gemini] ✅ Đã đổi sang key #{_gemini_key_idx + 1}")
    return True


def _rotate_groq_key() -> bool:
    """Chuyển sang Groq key tiếp theo."""
    global _groq_key_idx, groq_client, _groq_exhausted
    _groq_key_idx += 1
    if _groq_key_idx >= len(GROQ_API_KEYS):
        print("  [Groq] ⚠️ Đã dùng hết tất cả API key! Bỏ qua Groq cho các câu còn lại.")
        _groq_exhausted = True   # ← đặt cờ, không thử nữa
        return False
    groq_client = Groq(api_key=GROQ_API_KEYS[_groq_key_idx])
    print(f"  [Groq] ✅ Đã đổi sang key #{_groq_key_idx + 1}")
    return True


# ──────────────────────────────────────────────
#  HÀM GỌI CHATBOT LOCAL (OLLAMA)
# ──────────────────────────────────────────────
def ask_ollama(question: str, model_name: str, timeout: int = 120) -> str:
    """Gửi câu hỏi tới Ollama và nhận câu trả lời."""
    system_prompt = (
        "Bạn là trợ lý tư vấn tuyển sinh của Đại học Công nghiệp Hà Nội (HAUI). "
        "Hãy trả lời chính xác, thân thiện và ngắn gọn dựa trên thông tin tuyển sinh của trường."
    )
    payload = {
        "model":  model_name,
        "prompt": f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                  f"<|im_start|>user\n{question}<|im_end|>\n"
                  f"<|im_start|>assistant\n",
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()

        # Bỏ phần <think>...</think> nếu model Qwen3 trả về
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw if raw else "[Không có câu trả lời]"
    except Exception as e:
        return f"[LỖI OLLAMA: {e}]"


# ──────────────────────────────────────────────
#  PROMPT CHẤM ĐIỂM  (LLM-as-a-Judge)
# ──────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """
Bạn là giám khảo chuyên nghiệp đánh giá chất lượng câu trả lời của chatbot tư vấn tuyển sinh Đại học Công nghiệp Hà Nội (HAUI).

### Câu hỏi của người dùng:
{question}

### Câu trả lời CHUẨN (ground truth):
{ground_truth}

### Câu trả lời của CHATBOT cần đánh giá:
{bot_answer}

---
Hãy chấm điểm câu trả lời của chatbot theo 3 tiêu chí sau (thang điểm 1-5):

1. **Độ chính xác (Accuracy)**: Thông tin có đúng với câu trả lời chuẩn không?
   - 5: Hoàn toàn chính xác  
   - 4: Chủ yếu chính xác, sai sót nhỏ  
   - 3: Một nửa đúng  
   - 2: Phần lớn sai  
   - 1: Hoàn toàn sai hoặc không liên quan

2. **Độ liên quan (Relevance)**: Câu trả lời có đáp ứng đúng yêu cầu của câu hỏi không?
   - 5: Trả lời đúng trọng tâm, đầy đủ  
   - 3: Có trả lời nhưng thiếu thông tin  
   - 1: Lạc đề hoàn toàn

3. **Chất lượng ngôn ngữ (Fluency)**: Câu văn có tự nhiên, thân thiện, dễ hiểu không?
   - 5: Rất tự nhiên, thân thiện  
   - 3: Chấp nhận được  
   - 1: Cứng nhắc, khó hiểu

**Yêu cầu định dạng phản hồi** (chỉ trả về JSON, không thêm bất kỳ nội dung nào khác):
{{
  "accuracy": <số từ 1-5>,
  "relevance": <số từ 1-5>,
  "fluency": <số từ 1-5>,
  "comment": "<nhận xét ngắn gọn 1 câu bằng tiếng Việt>"
}}
"""


def parse_scores(text: str) -> dict:
    """Trích xuất điểm số từ phản hồi JSON của LLM."""
    try:
        # Thử tìm JSON trong chuỗi
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "accuracy":  float(data.get("accuracy",  0)),
                "relevance": float(data.get("relevance", 0)),
                "fluency":   float(data.get("fluency",   0)),
                "comment":   str(data.get("comment",    "")),
            }
    except Exception:
        pass
    return {"accuracy": 0, "relevance": 0, "fluency": 0, "comment": "Lỗi parse"}


# ──────────────────────────────────────────────
#  GỌI GEMINI CHẤM ĐIỂM  (Ỵ tự đổi key khi hết quota)
# ──────────────────────────────────────────────
DEAD_ERRORS  = ("CONSUMER_SUSPENDED", "PERMISSION_DENIED")
QUOTA_ERRORS = ("429", "RESOURCE_EXHAUSTED", "RATE_LIMIT")
RETRY_ERRORS = ("503", "UNAVAILABLE")   # quá tải tạm thời → chờ rồi thử lại

def judge_with_gemini(question: str, ground_truth: str, bot_answer: str) -> dict:
    global _gemini_client
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        bot_answer=bot_answer,
    )
    for attempt in range(len(GEMINI_API_KEYS) + 1):
        try:
            resp = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return parse_scores(resp.text)

        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in RETRY_ERRORS):
                # Gemini quá tải tạm thời → đợi 10s rồi thử lại cùng key
                print(f"  [Gemini] Server bận, chờ 10s rồi thử lại...")
                time.sleep(10)
                continue
            elif any(k in err_str for k in QUOTA_ERRORS + DEAD_ERRORS):
                print(f"  [Gemini] Key #{_gemini_key_idx + 1} hết quota, đổi key...")
                time.sleep(2)
                if not _rotate_gemini_key():
                    return judge_with_groq(question, ground_truth, bot_answer)
            else:
                print(f"  [Gemini lỗi]: {e}")
                return {"accuracy": 0, "relevance": 0, "fluency": 0, "comment": f"Lỗi: {e}"}

    return {"accuracy": 0, "relevance": 0, "fluency": 0, "comment": "Hết key Gemini"}


# ──────────────────────────────────────────────
#  GỌI GROQ CHẤM ĐIỂM  (Backup / Quan tòa 2)
# ──────────────────────────────────────────────
GROQ_QUOTA_ERRORS = ("429", "RESOURCE_EXHAUSTED", "RATE_LIMIT", "tokens", "quota")
GROQ_RETRY_ERRORS = ("503", "UNAVAILABLE", "high demand")

def judge_with_groq(question: str, ground_truth: str, bot_answer: str) -> dict:
    global groq_client
    # Nếu đã hết key từ trước → bỏ qua ngay không thử nữa
    if _groq_exhausted:
        return {"accuracy": 0, "relevance": 0, "fluency": 0, "comment": "Hết key Groq"}

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        bot_answer=bot_answer,
    )
    for attempt in range(len(GROQ_API_KEYS) + 1):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            return parse_scores(resp.choices[0].message.content)
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in GROQ_RETRY_ERRORS):
                print(f"  [Groq] Server bận, chờ 15s rồi thử lại...")
                time.sleep(15)
                continue
            elif any(k in err_str for k in GROQ_QUOTA_ERRORS):
                print(f"  [Groq] Key #{_groq_key_idx + 1} hết quota, đổi key...")
                if not _rotate_groq_key():
                    return {"accuracy": 0, "relevance": 0, "fluency": 0, "comment": "Hết key Groq"}
                time.sleep(3)
            else:
                print(f"  [Groq lỗi]: {e}")
                return {"accuracy": 0, "relevance": 0, "fluency": 0, "comment": f"Lỗi: {e}"}
    return {"accuracy": 0, "relevance": 0, "fluency": 0, "comment": "Hết key Groq"}


# ──────────────────────────────────────────────
#  MAIN — CHẠY ĐÁNH GIÁ FT1 vs FT2
# ──────────────────────────────────────────────
def score_answer(question, ground_truth, answer, label):
    """Chấm điểm 1 câu trả lời bằng cả 2 quan tòa, trả về dict"""
    print(f"  → Gemini chấm [{label}]...")
    g = judge_with_gemini(question, ground_truth, answer)
    time.sleep(1.5)

    print(f"  → Groq chấm [{label}]...")
    r = judge_with_groq(question, ground_truth, answer)
    time.sleep(0.5)

    # Tính điểm final (chỉ tính từ quan tòa có điểm > 0 — tránh lệch khi Groq hết key)
    judges = [s for s in [g, r] if s["accuracy"] > 0 or s["relevance"] > 0]
    if not judges:
        judges = [g]   # fallback không có điểm nào

    final_acc = round(sum(s["accuracy"] for s in judges) / len(judges), 2)
    final_rel = round(sum(s["relevance"] for s in judges) / len(judges), 2)
    final_flu = round(sum(s["fluency"]   for s in judges) / len(judges), 2)

    return {
        f"{label}_answer":      answer,
        f"{label}_Gemini_Acc":  g["accuracy"],
        f"{label}_Gemini_Rel":  g["relevance"],
        f"{label}_Gemini_Flu":  g["fluency"],
        f"{label}_Gemini_Avg":  round((g["accuracy"] + g["relevance"] + g["fluency"]) / 3, 2),
        f"{label}_Gemini_Note": g["comment"],
        f"{label}_Groq_Acc":    r["accuracy"],
        f"{label}_Groq_Rel":    r["relevance"],
        f"{label}_Groq_Flu":    r["fluency"],
        f"{label}_Groq_Avg":    round((r["accuracy"] + r["relevance"] + r["fluency"]) / 3, 2),
        f"{label}_Groq_Note":   r["comment"],
        f"{label}_Final_Acc":   final_acc,
        f"{label}_Final_Rel":   final_rel,
        f"{label}_Final_Flu":   final_flu,
        f"{label}_Final_Avg":   round((final_acc + final_rel + final_flu) / 3, 2),
    }


def main():
    print("=" * 65)
    print("  HAUI Chatbot — LLM-as-a-Judge  (FT1 Baseline vs FT2 Fine-tuned)")
    print("=" * 65)

    # 1. Load dữ liệu
    samples = []
    with open(VAL_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    samples = samples[:NUM_SAMPLES]
    print(f"\n✅ Đã load {len(samples)} mẫu | FT1={BASELINE_MODEL or 'OFF'} | FT2={HAUI_BOT_MODEL}")

    results = []
    for i, sample in enumerate(tqdm(samples, desc="Đang đánh giá")):
        question     = sample["instruction"]
        ground_truth = sample["output"]
        print(f"\n[{i+1}/{len(samples)}] {question[:65]}...")

        row = {"STT": i+1, "Câu hỏi": question, "Ground Truth": ground_truth}

        # ―― FT2: HAUI Bot (fine-tuned) ――
        print("  → Đang hỏi FT2 — HAUI Bot (fine-tuned)...")
        ft2_ans = ask_ollama(question, HAUI_BOT_MODEL)
        row.update(score_answer(question, ground_truth, ft2_ans, "FT2"))

        # ―― FT1: Baseline (chưa fine-tune) ――
        if BASELINE_MODEL:
            print(f"  → Đang hỏi FT1 — Baseline ({BASELINE_MODEL})...")
            ft1_ans = ask_ollama(question, BASELINE_MODEL)
            row.update(score_answer(question, ground_truth, ft1_ans, "FT1"))

        results.append(row)

    # 3. Lưu file
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Chi tiết", index=False)

        # ―― Sheet tóm tắt — chuẩn bảng báo cáo ――
        metrics = ["Accuracy", "Relevance", "Fluency", "Overall"]
        summary = {"Tiêu chí": ["Độ chính xác (Accuracy)",
                                   "Độ liên quan (Relevance)",
                                   "Chất lượng ngôn ngữ (Fluency)",
                                   "⭐ Điểm tổng thể"]}

        if BASELINE_MODEL and "FT1_Final_Acc" in df.columns:
            summary["FT1 — Baseline (avg)"] = [
                round(df["FT1_Final_Acc"].mean(), 3),
                round(df["FT1_Final_Rel"].mean(), 3),
                round(df["FT1_Final_Flu"].mean(), 3),
                round(df["FT1_Final_Avg"].mean(), 3),
            ]

        summary["FT2 — Fine-tuned (avg)"] = [
            round(df["FT2_Final_Acc"].mean(), 3),
            round(df["FT2_Final_Rel"].mean(), 3),
            round(df["FT2_Final_Flu"].mean(), 3),
            round(df["FT2_Final_Avg"].mean(), 3),
        ]

        if BASELINE_MODEL and "FT1_Final_Avg" in df.columns:
            delta = df["FT2_Final_Avg"].mean() - df["FT1_Final_Avg"].mean()
            summary["📈 Cải thiện (+/-)"] = [
                round(df["FT2_Final_Acc"].mean() - df["FT1_Final_Acc"].mean(), 3),
                round(df["FT2_Final_Rel"].mean() - df["FT1_Final_Rel"].mean(), 3),
                round(df["FT2_Final_Flu"].mean() - df["FT1_Final_Flu"].mean(), 3),
                round(delta, 3),
            ]

        pd.DataFrame(summary).to_excel(writer, sheet_name="Tóm tắt", index=False)

    # 4. In kết quả
    print("\n" + "=" * 65)
    print("  KẺT QUẢ ĐÁNH GIÁ")
    print("=" * 65)
    print(f"  Số mẫu : {len(df)}")
    print()
    if BASELINE_MODEL and "FT1_Final_Avg" in df.columns:
        print(f"  FT1 (Baseline)   Acc={df['FT1_Final_Acc'].mean():.3f}  Rel={df['FT1_Final_Rel'].mean():.3f}  Flu={df['FT1_Final_Flu'].mean():.3f}  ⭐={df['FT1_Final_Avg'].mean():.3f}")
    print(f"  FT2 (Fine-tuned) Acc={df['FT2_Final_Acc'].mean():.3f}  Rel={df['FT2_Final_Rel'].mean():.3f}  Flu={df['FT2_Final_Flu'].mean():.3f}  ⭐={df['FT2_Final_Avg'].mean():.3f}")
    if BASELINE_MODEL and "FT1_Final_Avg" in df.columns:
        delta = df['FT2_Final_Avg'].mean() - df['FT1_Final_Avg'].mean()
        sign  = "+" if delta >= 0 else ""
        print(f"  📈 Cải thiện sau Fine-tune: {sign}{delta:.3f}")

    print(f"\n✅ Đã lưu kết quả → {OUTPUT_EXCEL}")
    print(f"✅ Đã lưu CSV    → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

