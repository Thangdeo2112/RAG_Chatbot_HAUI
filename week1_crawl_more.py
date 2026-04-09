"""
TUẦN 1 — Crawl thêm data quan trọng
=====================================
Chạy xong script này → chạy lại step1 → step2 để reindex toàn bộ.

Chạy:
    cd RAG
    python week1_crawl_more.py
    python step1_prepare_chunks.py
    python step2_embed_index.py
"""

import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import trafilatura
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# ── 12 URL ĐÃ VERIFY — quan trọng nhất với thí sinh ─────────────────────────
CRAWL_TARGETS = [

    # ── NHÓM 1: ĐIỂM CHUẨN (thí sinh hỏi nhiều nhất) ──────────────────────
    {
        "url": "https://tuyensinh.haui.edu.vn/tin-tuc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2024/66c098473a85982920fb937f",
        "category": "diem_chuan",
        "description": "Điểm chuẩn & kết quả xét tuyển đại học chính quy 2024",
    },
    {
        "url": "https://tuyensinh.haui.edu.vn/diem-chuan-trung-tuyen-dai-hoc/ket-qua-xet-tuyen-dai-hoc-chinh-quy-nam-2023-theo-cac-phuong-thuc-2,-4,-5,-6-/649eb9a218495549ec635a3a",
        "category": "diem_chuan",
        "description": "Kết quả xét tuyển ĐH chính quy 2023 theo phương thức 2,4,5,6",
    },

    # ── NHÓM 2: ĐỀ ÁN & THÔNG TIN TUYỂN SINH TỪNG NĂM ─────────────────────
    {
        "url": "https://tuyensinh.haui.edu.vn/dai-hoc-chinh-quy/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2025/681c349af721616a54f64963",
        "category": "de_an_tuyen_sinh",
        "description": "Thông tin tuyển sinh đại học chính quy năm 2025 — đầy đủ nhất",
    },
    
]


# ── SCRAPER (tái sử dụng từ haui_crawler_v2) ─────────────────────────────────
class HauiScraper:
    def __init__(self, delay=2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
        })
        self.results = []

    def _get_html(self, url):
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except Exception as e:
            print(f"  [LỖI] {e}")
            return None

    def _extract_tables(self, html):
        soup = BeautifulSoup(html, "html.parser")
        tables = []
        for tbl in soup.find_all("table"):
            try:
                df = pd.read_html(str(tbl))[0].fillna("")
                if all(isinstance(c, int) for c in df.columns):
                    df.columns = df.iloc[0].astype(str).tolist()
                    df = df[1:].reset_index(drop=True)
                tables.append(df.to_dict(orient="records"))
            except Exception:
                continue
        return tables

    def _tables_to_prose(self, tables):
        lines = []
        for tbl in tables:
            for row in tbl:
                parts = [f"{k}: {v}" for k, v in row.items()
                         if str(v).strip() and str(k).strip()
                         and str(v) not in ("nan", "NaN")]
                if parts:
                    lines.append(", ".join(parts) + ".")
        return "\n".join(lines)

    def scrape(self, url, category, description=""):
        print(f"\n→ [{category}] {description}")
        html = self._get_html(url)
        if not html:
            return None

        extracted = trafilatura.extract(
            html, include_tables=True, include_links=False,
            output_format="json", with_metadata=True, favor_recall=True,
        )

        title, main_text = description, ""
        if extracted:
            d = json.loads(extracted)
            title = d.get("title") or description
            main_text = d.get("text") or ""

        tables_raw  = self._extract_tables(html)
        table_prose = self._tables_to_prose(tables_raw)

        full_text = main_text.strip()
        if table_prose:
            full_text += "\n\n[Dữ liệu bảng]\n" + table_prose

        if not full_text.strip():
            print("  [CẢNH BÁO] Không lấy được nội dung!")
            return None

        print(f"  [OK] {len(full_text):,} ký tự | {len(tables_raw)} bảng")
        return {
            "url"        : url,
            "title"      : title,
            "category"   : category,
            "description": description,
            "text_content": full_text,
            "tables_raw" : tables_raw,
            "char_count" : len(full_text),
            "table_count": len(tables_raw),
            "crawled_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def run(self, targets):
        for item in targets:
            rec = self.scrape(item["url"], item["category"],
                              item.get("description", ""))
            if rec:
                self.results.append(rec)
            if item != targets[-1]:
                time.sleep(self.delay)

        print(f"\n{'='*50}")
        print(f"✓ Hoàn thành: {len(self.results)}/{len(targets)} trang")
        total_chars = sum(r["char_count"] for r in self.results)
        print(f"✓ Tổng ký tự mới: {total_chars:,}")

    def save(self, rag_file="haui_week1_data.jsonl",
             debug_file="haui_week1_debug.jsonl"):
        # File RAG sạch (không có tables_raw)
        with open(rag_file, "w", encoding="utf-8") as f:
            for r in self.results:
                rec = {k: v for k, v in r.items()
                       if k not in ("tables_raw", "char_count", "table_count")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"✓ RAG data  → {rag_file}  ({len(self.results)} bản ghi)")

        # File debug đầy đủ
        with open(debug_file, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✓ Debug     → {debug_file}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("TUẦN 1 — Crawl thêm 12 URL quan trọng")
    print("=" * 50)

    scraper = HauiScraper(delay=2.0)
    scraper.run(CRAWL_TARGETS)

    if scraper.results:
        scraper.save()
        print("\n" + "=" * 50)
        print("BƯỚC TIẾP THEO:")
        print("  Mở step1_prepare_chunks.py")
        print("  Thêm 'haui_week1_data.jsonl' vào danh sách INPUT_FILES")
        print("  Rồi chạy: python step1_prepare_chunks.py")
        print("            python step2_embed_index.py")
    else:
        print("\n[LỖI] Không crawl được trang nào.")