# HAUI Admission AI Chatbot
(Trợ lý AI Tư vấn Tuyển sinh Đại học Công nghiệp Hà Nội - HAUI)

## 📌 Giới thiệu chung
Đây là dự án Đồ án tốt nghiệp nhằm xây dựng một hệ thống RAG (Retrieval-Augmented Generation) Chatbot hỗ trợ tư vấn thông tin tuyển sinh cho trường Đại học Công nghiệp Hà Nội. Hệ thống được tinh chỉnh (fine-tune) và tối ưu hóa để có thể chạy **hoàn toàn local (offline) trên CPU** của laptop cá nhân mà không cần đến GPU mạnh hay kết nối Internet.

## ✨ Các tính năng nổi bật
*   **Mô hình ngôn ngữ (LLM)**: Sử dụng base model **Qwen-3 8B** kết hợp phương pháp học tăng cường **LoRA Adapter**. Mô hình cũng đã được lượng tử hóa (Quantization) định dạng **GGUF (Q4_K_M)** giúp giảm dung lượng, phù hợp chạy trên CPU.
*   **RAG Engine Component**:
    *   **Embedding Model**: `BAAI/bge-m3` đa ngôn ngữ, hỗ trợ tiếng Việt xuất sắc.
    *   **Database**: Sử dụng Vector Database **ChromaDB** giúp lưu trữ và truy xuất ngữ cảnh chính xác, tốc độ cao.
*   **Xử lý và Suy luận (Inference)**: Tích hợp **Ollama** ở môi trường local, có tính năng phản hồi luồng (Async Streaming) tạo cảm giác mượt mà, phản hồi ngay lập tức như ChatGPT. 
*   **Giao diện người dùng (Gradio Custom UI)**: Được xây dựng bằng **Gradio 6.0** với thiết kế bố cục 3 cột chuyên nghiệp. Hỗ trợ giao diện bóng bẩy, responsive, ẩn avatar thừa và bong bóng chat được tối ưu CSS theo phong cách tối giản.

## ⚙️ Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu tiên quyết
*   Môi trường Python 3.9 trở lên.
*   Tải và cài đặt phần mềm [Ollama](https://ollama.com/) (đề chạy local LLM).

### 2. Cài đặt các thư viện cần thiết
Mở terminal/command prompt và cài đặt các dependencies chuẩn:
```bash
pip install chromadb sentence-transformers gradio requests torch
```

### 3. Thiết lập Model với Ollama
Đảm bảo bạn đã có file model định dạng GGUF (ví dụ `qwen3-8b.Q4_K_M.gguf`) và `Modelfile`. Khởi tạo model trong Ollama:
```bash
ollama create haui_bot -f Modelfile
```

### 4. Vận hành hệ thống theo từng bước
*Lưu ý: Bạn cần chạy lệnh trong đúng đường dẫn thư mục gốc của dự án này.*

**Bước 1: Làm sạch dữ liệu và tách đoạn (Chunks)**
Trích xuất dữ liệu thô (từ các nguồn crawl) và xử lý lại thành các tập tin JSONL chuẩn mực.
```bash
python step1_prepare_chunks.py
```

**Bước 2: Vector hóa & Lưu trữ thông tin (Embedding)**
Tạo index vào ChromaDB. Quá trình này sẽ mất một chút thời gian cho lần chạy đầu tiên.
```bash
python step2_embed_index.py
```
*Sau bước này, một thư mục `chroma_db/` sẽ được sinh ra ở máy của bạn.*

**Bước 3: Khởi chạy Trợ lý AI (Gradio Web UI)**
Chạy câu lệnh dưới và tận hưởng thành quả của bạn ở trình duyệt máy tính.
```bash
python step3_rag_chatbot.py
```
👉 Truy cập giao diện tại: **http://localhost:7860**

---

## 🛠 Cấu trúc thư mục
Dưới đây là một số thành phần quan trọng trong mã nguồn:
```text
RAG/
├── chroma_db/                  ← Thư mục CSDL vector sinh ra ở Bước 2
├── .gitignore                  ← Tệp cấu hình Git bỏ qua các file model nặng
├── Modelfile                   ← Tệp cấu hình của Ollama tạo custom model
├── qwen3-8b.Q4_K_M.gguf        ← File weights của LLM đã quantize 
├── step1_prepare_chunks.py     ← Mã nguồn tiền xử lý dữ liệu
├── step2_embed_index.py        ← Mã nguồn tạo RAG embeddings
├── step3_rag_chatbot.py        ← Mã nguồn khởi chạy Gradio UI & Stream Chatbot
...
```

## 📝 Chỉnh sửa & Cập nhật dữ liệu
*   Để **cập nhật thêm kiến thức mới**, bạn chỉ cần thêm các file JSONL / Raw text vào hệ thống, sau đó cấu hình ở mảng đọc data của mã nguồn `step1_prepare_chunks.py` và chạy lại từ **Bước 1**. 
*   Nếu muốn thay đổi Backend LLM sang OpenAI GPT hoặc Gemini thay vì Ollama, hãy chỉnh sửa đoạn khai báo `LLM_CALL` ở file `step3_rag_chatbot.py`.
