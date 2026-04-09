"""
TUẦN 2 — Sinh dataset Q&A rule-based (hoàn toàn offline)
==========================================================
Không cần Gemini, OpenAI hay bất kỳ LLM nào.
Đọc chunks → nhận dạng nội dung → ghép Q&A từ template.

Input : haui_chunks.jsonl
Output: haui_qa_dataset.jsonl, haui_qa_train.jsonl, haui_qa_val.jsonl

Chạy:
    python week2_generate_qa_offline.py
"""

import json
import random
import re
from pathlib import Path

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CHUNKS_FILE  = "haui_chunks.jsonl"
OUTPUT_FILE  = "haui_qa_dataset.jsonl"
TRAIN_FILE   = "haui_qa_train.jsonl"
VAL_FILE     = "haui_qa_val.jsonl"
TRAIN_RATIO  = 0.8

SYSTEM = (
    "Bạn là trợ lý tư vấn tuyển sinh của Đại học Công nghiệp Hà Nội (HAUI). "
    "Hãy trả lời chính xác, thân thiện và ngắn gọn dựa trên thông tin tuyển sinh của trường."
)

# ── Templates câu hỏi theo từng chủ đề ────────────────────────────────────────

# Câu hỏi chung về HAUI — dùng cho mọi chunk
GENERAL_QA = [
    ("Trường Đại học Công nghiệp Hà Nội còn được gọi là gì?",
     "Trường Đại học Công nghiệp Hà Nội còn được gọi tắt là HAUI (HaNoi University of Industry)."),
    ("HAUI là trường công lập hay tư thục?",
     "HAUI là trường đại học công lập, trực thuộc Bộ Công Thương."),
    ("Trường Đại học Công nghiệp Hà Nội ở đâu?",
     "Trường Đại học Công nghiệp Hà Nội có địa chỉ tại số 298 Cầu Diễn, quận Nam Từ Liêm, Hà Nội."),
    ("Hotline tư vấn tuyển sinh của HAUI là gì?",
     "Hotline tư vấn tuyển sinh của HAUI là 1900 558 826."),
    ("Website tuyển sinh của HAUI là gì?",
     "Website tuyển sinh chính thức của HAUI là tuyensinh.haui.edu.vn."),
    ("HAUI có bao nhiêu ngành đào tạo đại học chính quy?",
     "Năm 2026, HAUI tuyển sinh 72 ngành/chương trình đào tạo đại học chính quy."),
    ("HAUI tuyển bao nhiêu chỉ tiêu năm 2026?",
     "Năm 2026, HAUI dự kiến tuyển 8.300 chỉ tiêu đại học chính quy."),
]

# Template cho chunk chứa điểm chuẩn
DIEM_CHUAN_TEMPLATES = [
    "Điểm chuẩn ngành {nganh} của HAUI năm {nam} là bao nhiêu?",
    "Ngành {nganh} HAUI năm {nam} lấy bao nhiêu điểm?",
    "Muốn học ngành {nganh} tại HAUI năm {nam} cần bao nhiêu điểm?",
    "Điểm trúng tuyển ngành {nganh} HAUI năm {nam} là mấy điểm?",
    "Em cần đạt bao nhiêu điểm để vào ngành {nganh} của HAUI năm {nam}?",
]

# Template cho chunk chứa phương thức/tổ hợp
PHUONG_THUC_TEMPLATES = [
    "HAUI tuyển sinh theo những phương thức nào năm {nam}?",
    "Có mấy phương thức xét tuyển vào HAUI năm {nam}?",
    "Phương thức {pt} của HAUI xét dựa trên tiêu chí gì?",
    "Ngành {nganh} HAUI xét tuyển theo tổ hợp môn nào?",
    "Tổ hợp môn nào được dùng để xét tuyển ngành {nganh} tại HAUI?",
    "HAUI có xét tuyển bằng điểm thi đánh giá năng lực không?",
    "Thí sinh đoạt giải học sinh giỏi có được ưu tiên xét tuyển vào HAUI không?",
]

# Template cho chunk chứa lịch trình/kế hoạch
KE_HOACH_TEMPLATES = [
    "Thời gian đăng ký xét tuyển vào HAUI năm {nam} là khi nào?",
    "Khi nào HAUI công bố kết quả trúng tuyển đợt {dot}?",
    "Thí sinh xác nhận nhập học HAUI trước ngày nào?",
    "Lịch nhập học của HAUI năm {nam} như thế nào?",
    "HAUI bắt đầu học kỳ I năm {nam} vào ngày nào?",
    "Thời hạn nộp hồ sơ xét tuyển thẳng vào HAUI là khi nào?",
    "HAUI có mấy đợt xét tuyển trong năm?",
]

# Template cho chunk đề án tuyển sinh
DE_AN_TEMPLATES = [
    "HAUI có những ngành đào tạo mới nào năm {nam}?",
    "Chương trình đào tạo bằng tiếng Anh tại HAUI gồm những ngành gì?",
    "Điều kiện xét tuyển thẳng vào HAUI là gì?",
    "HAUI có chính sách ưu tiên gì cho thí sinh khu vực khó khăn?",
    "Ngành {nganh} tại HAUI có chỉ tiêu bao nhiêu?",
    "HAUI xét tuyển phương thức 2 dành cho đối tượng nào?",
    "Phương thức 4 xét tuyển dựa trên kết quả kỳ thi nào?",
    "HAUI có liên kết đào tạo quốc tế không?",
]


# ── Hàm trích xuất thông tin từ text ──────────────────────────────────────────

def extract_years(text):
    """Tìm các năm xuất hiện trong text."""
    years = re.findall(r'20(2[0-9])', text)
    return list(set(['20' + y for y in years])) or ['2026']


def extract_nganh_diem(text):
    """
    Trích xuất cặp (tên ngành, điểm chuẩn) từ chunk điểm chuẩn.
    Pattern: mã ngành | tên ngành | điểm
    """
    results = []
    # Pattern: 7xxxxxx | Tên ngành | 2x.xx
    pattern = r'7\d{6}\s*\|\s*([^|]{5,50}?)\s*\|\s*(2\d\.\d{1,2})'
    matches = re.findall(pattern, text)
    for nganh, diem in matches:
        nganh = nganh.strip()
        diem  = diem.strip()
        if len(nganh) > 3:
            results.append((nganh, diem))
    return results[:5]  # tối đa 5 ngành/chunk


def extract_nganh_tohop(text):
    """Trích xuất cặp (tên ngành, tổ hợp) từ chunk tổ hợp."""
    results = []
    # Pattern: tên ngành | Phương thức X; Tổ hợp YYY
    pattern = r'([A-ZĐÁÀẢÃẠĂẮẶẴẶẰÂẤẦẨẪẬÊẾỀỂỄỆÔỐỒỔỖỘƯỨỪỬỮỰÍÌỈĨỊÓÒỎÕỌÚÙỦŨỤÝỲỶỸỴ][^|]{3,40})\s*\|\s*(Phương thức[^|]{5,60}Tổ hợp[^|]{5,40})'
    matches = re.findall(pattern, text)
    for nganh, combo in matches:
        results.append((nganh.strip(), combo.strip()))
    return results[:4]


def extract_dates(text):
    """Trích xuất ngày tháng từ text."""
    dates = re.findall(r'\d{1,2}/\d{1,2}/20\d{2}', text)
    return dates[:3]


def extract_phuong_thuc_info(text):
    """Trích xuất mô tả các phương thức."""
    results = []
    pts = re.findall(r'[Pp]hương thức\s*(\d)\s*[:\-\(]?\s*([^.\n]{20,150})', text)
    for pt_num, mo_ta in pts:
        results.append((pt_num, mo_ta.strip()))
    return results[:5]


# ── Hàm sinh Q&A theo từng category ──────────────────────────────────────────

def make_qa(question, answer, category):
    """Tạo 1 bản ghi Q&A format Alpaca + ShareGPT."""
    return {
        "instruction"  : question,
        "input"        : "",
        "output"       : answer,
        "system"       : SYSTEM,
        "category"     : category,
        "conversations": [
            {"from": "system",    "value": SYSTEM},
            {"from": "human",     "value": question},
            {"from": "assistant", "value": answer},
        ],
    }


def gen_diem_chuan(chunk):
    qa_list = []
    text    = chunk["text"]
    years   = extract_years(text)
    nam     = years[0] if years else "2024"
    pairs   = extract_nganh_diem(text)

    for nganh, diem in pairs:
        # Sinh 2-3 câu hỏi khác nhau cho mỗi ngành
        templates = random.sample(DIEM_CHUAN_TEMPLATES, min(3, len(DIEM_CHUAN_TEMPLATES)))
        for tpl in templates:
            q = tpl.format(nganh=nganh, nam=nam)
            a = (f"Điểm chuẩn ngành {nganh} tại HAUI năm {nam} là {diem} điểm "
                 f"(xét theo kết quả thi tốt nghiệp THPT).")
            qa_list.append(make_qa(q, a, chunk["category"]))

    # Câu hỏi tổng quát về điểm chuẩn năm đó
    if nam:
        qa_list.append(make_qa(
            f"HAUI công bố điểm chuẩn năm {nam} vào thời điểm nào?",
            f"HAUI thông báo điểm chuẩn trúng tuyển đại học chính quy năm {nam} "
            f"sau khi Bộ GD&ĐT hoàn thành xét tuyển chung trên hệ thống, "
            f"thường vào tháng 8 hàng năm.",
            chunk["category"]
        ))

    return qa_list


def gen_phuong_thuc(chunk):
    qa_list = []
    text    = chunk["text"]
    years   = extract_years(text)
    nam     = years[0] if years else "2026"

    # Câu hỏi tổng quát về phương thức
    qa_list.append(make_qa(
        f"HAUI tuyển sinh theo những phương thức nào năm {nam}?",
        f"Năm {nam}, HAUI tuyển sinh theo 5 phương thức: "
        f"(1) Xét tuyển thẳng, "
        f"(2) Xét học sinh giỏi/chứng chỉ quốc tế kết hợp học bạ, "
        f"(3) Xét điểm thi tốt nghiệp THPT, "
        f"(4) Xét kết quả thi đánh giá năng lực ĐHQG Hà Nội, "
        f"(5) Xét kết quả thi đánh giá tư duy ĐH Bách khoa Hà Nội.",
        chunk["category"]
    ))

    # Câu hỏi về từng phương thức
    pt_infos = extract_phuong_thuc_info(text)
    for pt_num, mo_ta in pt_infos:
        qa_list.append(make_qa(
            f"Phương thức {pt_num} xét tuyển vào HAUI là gì?",
            f"Phương thức {pt_num} của HAUI: {mo_ta}",
            chunk["category"]
        ))

    # Câu hỏi về tổ hợp ngành
    nganh_tohop = extract_nganh_tohop(text)
    for nganh, combo in nganh_tohop:
        qa_list.append(make_qa(
            f"Ngành {nganh} tại HAUI xét tuyển theo tổ hợp môn nào?",
            f"Ngành {nganh} tại HAUI xét tuyển theo: {combo}.",
            chunk["category"]
        ))

    # Câu hỏi về số chỉ tiêu
    chi_tieu = re.search(r'(\d[\d\.]+)\s*chỉ tiêu', text)
    if chi_tieu:
        so = chi_tieu.group(1)
        qa_list.append(make_qa(
            f"HAUI tuyển bao nhiêu chỉ tiêu năm {nam}?",
            f"Năm {nam}, HAUI dự kiến tuyển {so} chỉ tiêu đại học chính quy.",
            chunk["category"]
        ))

    return qa_list


def gen_ke_hoach(chunk):
    qa_list = []
    text    = chunk["text"]
    years   = extract_years(text)
    nam     = years[0] if years else "2026"

    # Câu hỏi về lịch tuyển sinh
    qa_list.append(make_qa(
        f"Lịch tuyển sinh đại học chính quy của HAUI năm {nam} như thế nào?",
        f"HAUI triển khai tuyển sinh đại học chính quy năm {nam} theo kế hoạch "
        f"của Bộ GD&ĐT. Thí sinh đăng ký nguyện vọng trực tuyến trên hệ thống "
        f"tuyển sinh của Bộ GD&ĐT. Kết quả trúng tuyển được thông báo vào tháng 8.",
        chunk["category"]
    ))

    # Trích xuất ngày cụ thể
    dates = extract_dates(text)
    if dates:
        qa_list.append(make_qa(
            f"Thời hạn xác nhận nhập học tại HAUI năm {nam} là khi nào?",
            f"Thí sinh trúng tuyển cần xác nhận nhập học trực tuyến trên hệ thống "
            f"của Bộ GD&ĐT đúng thời hạn quy định. "
            f"Với HAUI năm {nam}, thí sinh cần hoàn thành xác nhận trước 17h00 "
            f"theo thông báo chính thức của trường.",
            chunk["category"]
        ))

    # Câu hỏi về liên thông
    if 'liên thông' in text.lower():
        qa_list.append(make_qa(
            "HAUI có đào tạo liên thông lên đại học không?",
            "Có, HAUI có chương trình đào tạo liên thông lên đại học chính quy "
            "dành cho sinh viên đã tốt nghiệp cao đẳng hoặc trung cấp nghề. "
            "Thông tin chi tiết xem tại tuyensinh.haui.edu.vn.",
            chunk["category"]
        ))

    # Câu hỏi nhập học
    if 'nhập học' in text.lower():
        qa_list.append(make_qa(
            f"Sinh viên mới của HAUI bắt đầu học kỳ I năm {nam} vào ngày nào?",
            f"Theo kế hoạch tuyển sinh năm {nam}, sinh viên mới HAUI bắt đầu "
            f"học kỳ I vào tháng 9 sau khi hoàn thành tuần sinh hoạt công dân "
            f"và các thủ tục nhập học.",
            chunk["category"]
        ))

    return qa_list


def gen_de_an(chunk):
    qa_list = []
    text    = chunk["text"]
    years   = extract_years(text)
    nam     = years[0] if years else "2025"

    # Câu hỏi về ngành mới
    nganh_moi_match = re.findall(
        r'(Công nghệ sinh học|Trí tuệ nhân tạo|Vi mạch bán dẫn|'
        r'Công nghệ vật liệu|Kỹ thuật ô tô và năng lượng mới)', text
    )
    if nganh_moi_match:
        nganh_list = list(set(nganh_moi_match))
        qa_list.append(make_qa(
            f"HAUI có những ngành đào tạo mới nào năm {nam}?",
            f"Năm {nam}, HAUI mở thêm các ngành đào tạo mới gồm: "
            + ", ".join(nganh_list) + ".",
            chunk["category"]
        ))

    # Câu hỏi về chương trình tiếng Anh
    if 'tiếng Anh' in text or 'Tiếng Anh' in text:
        so_chuong_trinh = re.search(r'(\d+)\s*chương trình đào tạo bằng [Tt]iếng Anh', text)
        so = so_chuong_trinh.group(1) if so_chuong_trinh else "một số"
        qa_list.append(make_qa(
            f"HAUI có chương trình đào tạo bằng tiếng Anh không?",
            f"Có, HAUI có {so} chương trình đào tạo bằng tiếng Anh (chương trình "
            f"chất lượng cao). Sinh viên học toàn bộ bằng tiếng Anh, "
            f"được cấp bằng quốc tế và có cơ hội học tập/trao đổi tại nước ngoài.",
            chunk["category"]
        ))

    # Câu hỏi về xét tuyển thẳng
    if 'xét tuyển thẳng' in text.lower():
        qa_list.append(make_qa(
            "Điều kiện được xét tuyển thẳng vào HAUI là gì?",
            "Thí sinh được xét tuyển thẳng vào HAUI (Phương thức 1) nếu thuộc "
            "các đối tượng theo quy chế của Bộ GD&ĐT: học sinh giỏi quốc gia, "
            "quốc tế, thí sinh đặc cách theo quy định. "
            "Hồ sơ nộp trực tiếp về Phòng Đào tạo HAUI.",
            chunk["category"]
        ))

    # Câu hỏi về chỉ tiêu ngành cụ thể
    nganh_chitieu = re.findall(
        r'([A-ZĐÁÀẢÃẠĂẮẶẴẶẰÂẤẦẨẪẬÊẾỀỂỄỆÔỐỒỔỖỘƯỨỪỬỮỰÍÌỈĨỊÓÒỎÕỌÚÙỦŨỤÝỲỶỸỴ]'
        r'[a-zđáàảãạăắặẵặằâấầẩẫậêếềểễệôốồổỗộưứừửữựíìỉĩịóòỏõọúùủũụýỳỷỹỵ\s]{5,30})'
        r'\s*\|\s*\d+\s*\|\s*([A-Z]\d{2})',
        text
    )
    for nganh, tohop in nganh_chitieu[:3]:
        nganh = nganh.strip()
        qa_list.append(make_qa(
            f"Ngành {nganh} tại HAUI xét tuyển tổ hợp gì?",
            f"Ngành {nganh} tại HAUI xét tuyển theo tổ hợp {tohop} "
            f"và một số tổ hợp khác theo đề án tuyển sinh của trường. "
            f"Xem chi tiết tại tuyensinh.haui.edu.vn.",
            chunk["category"]
        ))

    # Câu hỏi chung về đề án
    qa_list.append(make_qa(
        f"Đề án tuyển sinh HAUI năm {nam} có gì mới so với các năm trước?",
        f"Đề án tuyển sinh HAUI năm {nam} có một số điểm mới: "
        f"bổ sung ngành đào tạo mới, điều chỉnh chỉ tiêu một số ngành, "
        f"và duy trì 5 phương thức xét tuyển. "
        f"Thông tin chi tiết xem tại tuyensinh.haui.edu.vn.",
        chunk["category"]
    ))

    return qa_list


# ── Dispatcher ────────────────────────────────────────────────────────────────

GENERATORS = {
    "diem_chuan"        : gen_diem_chuan,
    "phuong_thuc_to_hop": gen_phuong_thuc,
    "ke_hoach_tuyen_sinh": gen_ke_hoach,
    "de_an_tuyen_sinh"  : gen_de_an,
}


def process_chunk(chunk):
    cat = chunk.get("category", "")
    gen = GENERATORS.get(cat)
    if gen:
        return gen(chunk)
    return []


# ── Lưu file ──────────────────────────────────────────────────────────────────

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(data):>4} bản ghi → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("TUẦN 2 — Sinh Q&A rule-based (offline, không cần LLM)")
    print("=" * 55)

    if not Path(CHUNKS_FILE).exists():
        print(f"[LỖI] Không tìm thấy {CHUNKS_FILE}")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f if l.strip()]
    print(f"\n✓ Tải {len(chunks)} chunks")

    # Sinh Q&A từ chunks
    all_qa = []
    cat_stats = {}

    for chunk in chunks:
        pairs = process_chunk(chunk)
        all_qa.extend(pairs)
        cat = chunk.get("category", "unknown")
        cat_stats[cat] = cat_stats.get(cat, 0) + len(pairs)

    # Thêm Q&A chung về HAUI
    for q, a in GENERAL_QA:
        all_qa.append(make_qa(q, a, "general"))
    cat_stats["general"] = len(GENERAL_QA)

    # Dedup theo cặp (hỏi + 30 ký tự đầu trả lời)
    # → giữ lại câu hỏi giống nhau nhưng câu trả lời khác (ngành/năm khác)
    seen_qa = set()
    deduped = []
    for item in all_qa:
        key = item["instruction"].strip().lower() + "|" + item["output"][:30].lower()
        if key not in seen_qa:
            seen_qa.add(key)
            deduped.append(item)
    removed = len(all_qa) - len(deduped)
    all_qa  = deduped

    # Shuffle và split
    random.seed(42)
    random.shuffle(all_qa)
    split      = int(len(all_qa) * TRAIN_RATIO)
    train_data = all_qa[:split]
    val_data   = all_qa[split:]

    print(f"\nKết quả:")
    for cat, cnt in sorted(cat_stats.items(), key=lambda x: -x[1]):
        print(f"  {cnt:4d} Q&A | {cat}")
    if removed:
        print(f"  {removed:4d} bị xoá (trùng câu hỏi)")

    print(f"\nLưu file:")
    save_jsonl(all_qa,     OUTPUT_FILE)
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(val_data,   VAL_FILE)

    print(f"\n{'='*55}")
    print(f"THỐNG KÊ CUỐI:")
    print(f"  Tổng Q&A  : {len(all_qa)}")
    print(f"  Train     : {len(train_data)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val       : {len(val_data)} ({(1-TRAIN_RATIO)*100:.0f}%)")

    # Đánh giá
    print()
    if len(all_qa) >= 300:
        print("✓ Dataset ĐỦ để fine-tune (>= 300 cặp)")
    elif len(all_qa) >= 150:
        print("⚠ Dataset tối thiểu (150-299 cặp) — fine-tune được nhưng chất lượng vừa")
    else:
        print("✗ Dataset ít quá — xem lại chunks hoặc mở rộng templates")

    # In 3 mẫu ngẫu nhiên
    print("\n--- 3 mẫu Q&A ngẫu nhiên ---")
    for s in random.sample(all_qa, min(3, len(all_qa))):
        print(f"\n[{s['category']}]")
        print(f"Q: {s['instruction']}")
        print(f"A: {s['output'][:150]}{'...' if len(s['output'])>150 else ''}")

    print(f"\n✓ Xong! Upload {TRAIN_FILE} và {VAL_FILE} lên Kaggle để fine-tune.")


if __name__ == "__main__":
    main()
