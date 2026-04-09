# RAG Chatbot Tuyển sinh HAUI
## Cài đặt & Chạy

### 1. Cài thư viện
```bash
pip install chromadb sentence-transformers google-generativeai gradio
```

### 2. Lấy Gemini API Key (miễn phí)
Vào https://aistudio.google.com/app/apikey → tạo key → copy

### 3. Chạy theo thứ tự

```bash
# Bước 1: Làm sạch & cắt chunks từ 3 file JSONL
python step1_prepare_chunks.py

# Bước 2: Embed & lưu vào ChromaDB (~5-10 phút lần đầu)
python step2_embed_index.py

# Bước 3: Chạy chatbot
GEMINI_API_KEY=your_key_here python step3_rag_chatbot.py
```

Truy cập: http://localhost:7860

---

### Cấu trúc thư mục sau khi chạy
```
haui_rag/
├── haui_rag_raw_data.jsonl     ← file gốc 1
├── haui_rag_data_v2.jsonl      ← file gốc 2
├── haui_debug_v2.jsonl         ← file gốc 3
├── haui_chunks.jsonl           ← output bước 1
├── chroma_db/                  ← output bước 2 (vector DB)
├── step1_prepare_chunks.py
├── step2_embed_index.py
└── step3_rag_chatbot.py
```

### Đổi LLM (nếu không dùng Gemini)
Trong `step3_rag_chatbot.py`, uncomment phần `Backend 2` (OpenAI)
hoặc `Backend 3` (Ollama local) và comment lại phần Gemini.

### Thêm dữ liệu mới
Crawl thêm URL → thêm file JSONL vào `INPUT_FILES` trong bước 1
→ chạy lại cả 3 bước.
