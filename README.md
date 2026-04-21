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
*   **Đánh giá chất lượng tự động (LLM-as-a-Judge)**: Tích hợp framework đánh giá khách quan sử dụng API từ **Google Gemini** và **Groq Llama-3**. Chấm điểm trên 3 tiêu chí: Độ chính xác (Accuracy), Độ liên quan (Relevance), và Chất lượng ngôn ngữ (Fluency) để so sánh khách quan phiên bản gốc (Baseline) và bản tinh chỉnh (Fine-tuned).

## ⚙️ Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu tiên quyết
*   Môi trường Python 3.9 trở lên.
*   Tải và cài đặt phần mềm [Ollama](https://ollama.com/) (để chạy local LLM).

### 2. Cài đặt các thư viện cần thiết
Mở terminal/command prompt và cài đặt các dependencies chuẩn:
```bash
pip install chromadb sentence-transformers gradio requests torch
pip install google-generativeai groq openpyxl pandas tqdm python-docx pypdf2
```

### 3. Thiết lập Model với Ollama
Đảm bảo bạn đã có file model định dạng GGUF (ví dụ `qwen3-8b.Q4_K_M.gguf`) và cấu hình `Modelfile`. Sau đó, khởi tạo model trong Ollama:
```bash
ollama create haui_bot -f Modelfile
```

### 4. Vận hành hệ thống theo từng bước
*Lưu ý: Bạn cần chạy lệnh trong đúng đường dẫn thư mục gốc của dự án này.*

**Bước 1: Làm sạch dữ liệu và tách đoạn (Chunks)**
Trích xuất dữ liệu thô (từ các nguồn file pdf, docx qua `read_pdf.py` / `read_docx.py` hoặc crawl web) và xử lý lại thành các tập tin JSONL chuẩn mực.
```bash
python step1_prepare_chunks.py
```

**Bước 2: Vector hóa & Lưu trữ thông tin (Embedding)**
Tạo index vào vector database ChromaDB. Quá trình này sẽ mất một chút thời gian cho lần chạy đầu tiên.
```bash
python step2_embed_index.py
```
*Sau bước này, một thư mục `chroma_db/` sẽ được sinh ra ở máy của bạn.*

**Bước 3: Khởi chạy Trợ lý AI (Gradio Web UI)**
Chạy câu lệnh dưới và tận dụng thành quả của bạn ở trình duyệt:
```bash
python step3_rag_chatbot.py
```
👉 Truy cập giao diện tại: **http://localhost:7860**

**Bước 4: Đánh giá mô hình bằng LLM-as-a-Judge (Tùy chọn)**
Nếu bạn muốn kiểm tra, chạy thử nghiệm hiệu năng thực tế của model sau khi fine-tune so với bản gốc:
```bash
python evaluate_llm_judge.py
```
Sau quá trình tự động đánh giá, hệ thống xuất ra các file Excel và CSV chi tiết điểm số. Để xem tóm tắt kết quả hiệu suất một cách nhanh chóng, hãy chạy:
```bash
python check_result.py
```

---

## 🛠 Cấu trúc thư mục
Dưới đây là một số thành phần quan trọng trong mã nguồn dự án:
```text
RAG/
├── chroma_db/                  ← Thư mục Database Vector sinh ra ở Bước 2
├── qwen3_haui_lora/            ← Thư mục chứa trọng số LoRA adapter sau khi huấn luyện
├── .gitignore                  ← Tệp cấu hình, sẽ bỏ qua các models có dung lượng lớn
├── Modelfile                   ← Tệp lệnh thao tác của Ollama để tạo custom chatbot
├── fine-tune-haui-1.ipynb      ← Noteook minh hoạ quá trình Fine-tune tham số
├── evaluate_llm_judge.py       ← Chạy đánh giá độ chính xác sử dụng LLM-as-a-Judge (Gemini/Groq)
├── check_result.py             ← Script đọc và tóm tắt kết quả file Evaluate ra màn hình
├── read_docx.py / read_pdf.py  ← Code hỗ trợ đọc khối nội dung từ các file quy chế (.doc, .pdf)
├── qwen3-8b.Q4_K_M.gguf        ← File weights (quantize) của model LLM 
├── step1_prepare_chunks.py     ← Mã nguồn tiền xử lý và cắt đoạn văn bản dữ liệu
├── step2_embed_index.py        ← Mã nguồn embedding tài liệu vào ChromaDB
├── step3_rag_chatbot.py        ← Khởi chạy server giao diện Web & kết nối inference model
├── haui_qa_val.jsonl           ← Tập dữ liệu kiểm thử validation để đánh giá
...
```

## 📝 Chỉnh sửa & Cập nhật dữ liệu
*   Để **cập nhật thêm kiến thức mới**, bạn chỉ cần tạo dữ liệu bằng các reader (eg. `read_docx.py`) và thêm file JSONL / Raw text vào hệ thống, sau đó quy định lại đường dẫn ở mảng đọc data của mã nguồn `step1_prepare_chunks.py` và chạy lại các luồng xử lý từ **Bước 1**. 
*   Nếu muốn thay đổi Backend LLM sang OpenAI GPT hoặc Gemini thay vì Ollama, hãy chỉnh sửa logic của hàm request `LLM_CALL` ở file `step3_rag_chatbot.py`.
