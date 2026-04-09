"""
BƯỚC 3 — RAG Chatbot (Gradio UI với Streaming)

Cài đặt:
    pip install chromadb sentence-transformers gradio requests
"""

import json
import os
import re
from pathlib import Path

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR    = os.path.join(CURRENT_DIR, "chroma_db")
COLLECTION    = "haui_admission"
EMBED_MODEL   = "BAAI/bge-m3"
TOP_K         = 5
MAX_CTX_CHARS = 3000

SYSTEM_PROMPT = """Bạn là trợ lý tư vấn tuyển sinh của Đại học Công nghiệp Hà Nội (HAUI).
Nhiệm vụ của bạn là trả lời chính xác, ngắn gọn và thân thiện các câu hỏi về tuyển sinh.

Nguyên tắc:
- Chỉ trả lời dựa trên thông tin trong phần [NGỮ CẢNH] bên dưới.
- Nếu không tìm thấy thông tin, hãy nói "Tôi chưa có thông tin về vấn đề này, bạn vui lòng liên hệ phòng tuyển sinh HAUI để được hỗ trợ."
- Không bịa đặt số liệu (điểm chuẩn, học phí, chỉ tiêu...).
- Trả lời bằng tiếng Việt, thân thiện, dễ hiểu."""

RAG_PROMPT_TEMPLATE = """{system}

[NGỮ CẢNH]
{context}

[CÂU HỎI]
{question}

[TRẢ LỜI]"""


# ── LLM BACKEND: Ollama Local ─────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    """Gọi Ollama, không stream. Dùng cho testing."""
    import requests
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model"   : "haui_bot",
        "messages": [{"role": "user", "content": prompt}],
        "stream"  : False,
        "think"   : False,
        "options" : {"num_predict": 300, "num_ctx": 2048},
    }, timeout=180)
    return resp.json().get("message", {}).get("content", "")

LLM_CALL = call_ollama


# ── RAG ENGINE ────────────────────────────────────────────────────────────────
class HauiRAG:
    def __init__(self):
        print("Đang tải RAG engine...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(COLLECTION)
        print(f"✓ RAG engine sẵn sàng — {self.collection.count()} vectors trong DB")

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        q_vec = self.embedder.encode(question, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text"      : doc,
                "category"  : meta.get("category", ""),
                "url"       : meta.get("url", ""),
                "title"     : meta.get("title", ""),
                "similarity": round(1 - dist, 3),
            })
        return chunks

    def build_context(self, chunks: list[dict]) -> str:
        parts, total = [], 0
        for c in chunks:
            txt = f"[{c['category']}] {c['text']}"
            if total + len(txt) > MAX_CTX_CHARS:
                break
            parts.append(txt)
            total += len(txt)
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str) -> tuple[str, list[dict]]:
        if not question.strip():
            return "Bạn ơi, bạn muốn hỏi gì về tuyển sinh HAUI vậy?", []
        chunks  = self.retrieve(question)
        context = self.build_context(chunks)
        prompt  = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT, context=context, question=question
        )
        try:
            answer = LLM_CALL(prompt)
            answer = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL).strip()
        except Exception as e:
            answer = f"[LỖI gọi LLM] {e}"
        return answer, chunks


# ── GRADIO UI (3 CỘT, STREAMING THẬT, CSS KIỂU CHATGPT) ──────────────────────
def launch_ui(rag: HauiRAG):
    try:
        import gradio as gr
        import requests
    except ImportError:
        print("Thiếu gradio. Chạy: pip install gradio")
        return

    def chat_stream(question: str, history: list):
        """Generator: yields (textbox_clear, updated_history) sau mỗi token."""
        if not question.strip():
            yield "", history
            return

        # RAG: embed → retrieve → build context
        chunks  = rag.retrieve(question)
        context = rag.build_context(chunks)
        prompt  = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT, context=context, question=question
        )

        # Gradio 6: bắt buộc dùng dict format {role, content}
        history = history + [{"role": "user", "content": question}]
        history.append({"role": "assistant", "content": "..."})
        yield "", history

        # Gọi Ollama stream từng token
        full_text = ""
        in_think  = False
        try:
            resp = requests.post("http://localhost:11434/api/chat", json={
                "model"   : "haui_bot",
                "messages": [{"role": "user", "content": prompt}],
                "stream"  : True,
                "think"   : False,
                "options" : {"num_predict": 512, "num_ctx": 2048, "temperature": 0.7},
            }, stream=True, timeout=180)

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    word = data.get("message", {}).get("content", "")
                    if not word:
                        continue
                    # Ẩn hoàn toàn khối <think>…</think>
                    if "<think>" in word:
                        in_think = True
                    if "</think>" in word:
                        in_think = False
                        continue
                    if in_think:
                        continue

                    full_text += word
                    history[-1]["content"] = full_text  # cập nhật nội dung trả lời
                    yield "", history
                except Exception:
                    pass

        except Exception as e:
            history[-1]["content"] = f"⚠️ Lỗi kết nối Ollama: {e}"
            yield "", history
            return

        # Làm sạch lần cuối phòng khi còn sót thẻ <think>
        clean = re.sub(r'<think>.*?</think>\s*', '', full_text, flags=re.DOTALL).strip()
        history[-1]["content"] = clean or full_text
        yield "", history


    # ── CSS tùy chỉnh 3 cột trắng/xám/đen giống ChatGPT ──────────────────────
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    body, .gradio-container * { font-family: 'Inter', sans-serif !important; }

    .gradio-container {
        background: #FFFFFF !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    footer { display: none !important; }

    /* ─── LAYOUT 3 CỘT ─── */
    #main-row { gap: 0 !important; min-height: 100vh; align-items: stretch !important; }

    /* Sidebar trái */
    #sidebar {
        background: #F9F9F9 !important;
        border-right: 1px solid #E5E7EB !important;
        padding: 28px 18px !important;
        overflow-y: auto;
    }
    #sidebar .prose { font-size: 0.84rem !important; color: #374151 !important; line-height: 1.6; }
    #sidebar .prose h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #111 !important; }
    #sidebar .prose p  { color: #6B7280 !important; margin: 2px 0 !important; }
    #sidebar .prose hr { border-color: #E5E7EB !important; }

    /* Cột giữa */
    #chat-col { padding: 0 !important; display: flex; flex-direction: column; }

    /* Chatbot - ẩn hoàn toàn avatar/icon thừa */
    #chatbot { border: none !important; box-shadow: none !important; background: transparent !important; }

    /* Ẩn mọi ô avatar - thử tất cả selector có thể */
    #chatbot .avatar-container,
    #chatbot .avatar,
    #chatbot [class*="avatar"],
    #chatbot [class*="Avatar"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        flex: none !important;
        min-width: 0 !important;
    }

    /* Ẩn nút copy/like/dislike */
    #chatbot .message-buttons,
    #chatbot [class*="message-button"],
    #chatbot [class*="copy"],
    #chatbot [class*="feedback"] { display: none !important; }

    /* Hàng chứa mỗi tin nhắn */
    #chatbot .message-row {
        display: flex !important;
        padding: 6px 20px !important;
        gap: 0 !important;
        align-items: flex-start !important;
        width: 100% !important;
    }

    /* USER bubble — xám nhạt, canh phải */
    #chatbot .message-row.user-row { justify-content: flex-end !important; }
    #chatbot .message-row.user-row .message {
        background: #F3F4F6 !important;
        color: #111 !important;
        border-radius: 18px 18px 4px 18px !important;
        border: none !important;
        max-width: 80% !important;
        width: auto !important;
        flex-grow: 0 !important;
        padding: 12px 18px !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        text-align: left !important;
    }
    #chatbot .message-row.user-row .message * {
        margin: 0 !important;
        width: auto !important;
        max-width: 100% !important;
        white-space: normal !important;
    }

    /* BOT — không nền, canh trái */
    #chatbot .message-row.bot-row { justify-content: flex-start !important; }
    #chatbot .message-row.bot-row .message {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #111 !important;
        max-width: 88% !important;
        width: auto !important;
        flex-grow: 0 !important;
        padding: 4px 0 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }


    /* Input row */
    #input-row { padding: 6px 20px 16px !important; border-top: 1px solid #F1F5F9 !important; }
    #txt_in textarea {
        border-radius: 26px !important;
        border: 1px solid #E5E7EB !important;
        background: #F9F9F9 !important;
        padding: 12px 18px !important;
        font-size: 0.95rem !important;
        resize: none !important;
        min-height: 44px !important;
    }
    #txt_in textarea:focus { border-color: #9CA3AF !important; background: #fff !important; outline: none !important; }

    /* Nút Gửi — tròn đen */
    #btn-send {
        background: #111 !important;
        border-radius: 50% !important;
        width: 44px !important;
        min-width: 44px !important;
        max-width: 44px !important;
        height: 44px !important;
        border: none !important;
        color: white !important;
        padding: 0 !important;
    }
    #btn-send:hover { background: #374151 !important; }

    /* Nút gợi ý (suggestion tags) */
    .suggest-row { padding: 4px 20px 8px !important; gap: 6px !important; flex-wrap: wrap !important; }
    .suggest-row button {
        background: #F9F9F9 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 20px !important;
        color: #374151 !important;
        font-size: 0.8rem !important;
        padding: 4px 12px !important;
        cursor: pointer !important;
        white-space: nowrap !important;
        height: auto !important;
        min-height: 28px !important;
        box-shadow: none !important;
    }
    .suggest-row button:hover { background: #F1F5F9 !important; border-color: #9CA3AF !important; }

    /* Nút Xóa */
    #btn-clear {
        background: transparent !important;
        color: #9CA3AF !important;
        border: none !important;
        font-size: 0.8rem !important;
        box-shadow: none !important;
        padding: 2px 8px !important;
        min-width: 80px !important;
    }

    /* Sidebar phải */
    #sidebar-right {
        background: #FFFFFF !important;
        border-left: 1px solid #E5E7EB !important;
        padding: 40px 14px !important;
        text-align: center !important;
    }
    #sidebar-right .prose { font-size: 0.82rem !important; color: #9CA3AF !important; }
    #sidebar-right .prose h3 { font-size: 0.9rem !important; color: #374151 !important; margin: 8px 0 !important; }
    """


    examples = [
        "HAUI tuyển sinh theo những phương thức nào năm 2026?",
        "Ngành CNTT xét tổ hợp môn gì?",
        "Chỉ tiêu tuyển sinh đại học chính quy năm 2026?",
        "Thời hạn đăng ký xét tuyển là khi nào?",
    ]

    with gr.Blocks(title="HAUI Tuyển Sinh — AI Chatbot Đồ Án") as demo:

        with gr.Row(elem_id="main-row"):

            # ─ CỘT TRÁI (scale=2)
            with gr.Column(scale=2, elem_id="sidebar"):
                gr.Markdown("""
### Đề tài: HAUI Tuyển Sinh
Trợ lý AI tự động giải đáp thông tin tuyển sinh Đại học Công nghiệp Hà Nội.

---

**Mô hình**
Base: Qwen-3 8B
Tích hợp: LoRA Adaptor

**Cấu trúc**
Kiến trúc RAG + Suy luận chuỗi, chạy Local không cần Internet.
                """)

            # ─ CỘT GIỮA (scale=8, chiếm phần lớn)
            with gr.Column(scale=8, elem_id="chat-col"):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Xin chào! Tôi là Trợ lý AI tuyển sinh HAUI. Bạn muốn hỏi gì về kỳ tuyển sinh sắp tới?"}],
                    elem_id="chatbot",
                    show_label=False,
                    height=580,
                )
                with gr.Row(elem_id="input-row"):
                    txt_in = gr.Textbox(
                        placeholder="Nhắn tin cho HAUI...",
                        show_label=False,
                        scale=9,
                        elem_id="txt_in",
                        lines=1,
                    )
                    btn_send = gr.Button("↑", scale=1, variant="primary", elem_id="btn-send")

                # Gợi ý câu hỏi dạng tag nhỏ gọn (thay gr.Examples)
                with gr.Row(elem_classes=["suggest-row"]):
                    btn_clear = gr.Button("↩ Xóa", elem_id="btn-clear")
                    q1 = gr.Button("Phương thức tuyển sinh 2026?")
                    q2 = gr.Button("Ngành CNTT xét tổ hợp gì?")
                    q3 = gr.Button("Chỉ tiêu tuyển sinh 2026?")
                    q4 = gr.Button("Hạn đăng ký là khi nào?")

            # ─ CỘT PHẢI (scale=1, nhỏ gọn)
            with gr.Column(scale=1, elem_id="sidebar-right"):
                gr.Markdown("""
<div style="font-size:3.5rem;margin-bottom:10px">🎓</div>

### Cố vấn AI

Sẵn sàng 24/7

❤️
                """)

        # ─ Sự kiện
        btn_send.click(chat_stream, [txt_in, chatbot], [txt_in, chatbot])
        txt_in.submit(chat_stream, [txt_in, chatbot], [txt_in, chatbot])
        btn_clear.click(lambda: [], outputs=chatbot)
        q1.click(lambda: "HAUI tuyển sinh theo những phương thức nào năm 2026?", outputs=txt_in)
        q2.click(lambda: "Ngành CNTT xét tổ hợp môn gì?", outputs=txt_in)
        q3.click(lambda: "Chỉ tiêu tuyển sinh đại học chính quy năm 2026?", outputs=txt_in)
        q4.click(lambda: "Thời hạn đăng ký xét tuyển là khi nào?", outputs=txt_in)

    print("\n🎉 GIAO DIỆN CHATBOT ĐỒ ÁN: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)




# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("BƯỚC 3 — RAG Chatbot (Gradio + Ollama Stream)")
    print("=" * 55)

    if not Path(CHROMA_DIR).exists():
        print(f"[LỖI] Không tìm thấy '{CHROMA_DIR}/'")
        print("Hãy chạy bước 2 trước: python step2_embed_index.py")
        exit(1)

    rag = HauiRAG()

    print("\nKhởi động Gradio Chatbot...")
    launch_ui(rag)