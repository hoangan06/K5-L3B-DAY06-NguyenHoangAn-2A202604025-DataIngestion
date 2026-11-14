"""Đọc và chuẩn hóa văn bản pháp luật (DOCX/HTML) thành các đơn vị Điều.

Kế thừa và chỉnh lại từ Project 4 (RAG văn bản pháp quy FMCG) của tác giả:
``read_docx_paragraphs``, ``read_html_paragraphs`` và logic tách Chương/Điều của
``chunk_paragraphs``. Thay đổi chính so với bản gốc: chuẩn hóa Unicode NFC, trả
về ``DocumentPage`` theo contract của starter, ghi log lỗi thay vì ``print`` và
không tự điền nội dung cho Điều rỗng (Điều bị bãi bỏ được ghi log và loại khỏi
phần nội dung).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .models import DocumentPage

PATTERN_CHUONG = re.compile(r"^Chương\s+([IVXLCDM]+)\b")
PATTERN_DIEU = re.compile(r"^Điều\s+(\d+)\.\s*(.*)$")
PATTERN_KET_THUC = re.compile(r"^Nơi nhận\s*:", re.IGNORECASE)
PATTERN_MUC = re.compile(r"^Mục\s+(\d+)\s*$")


def is_heading_line(para: str) -> bool:
    """Dòng tiêu đề Chương/Mục viết HOA toàn bộ (ví dụ 'ĐỐI VỚI THỰC PHẨM NHẬP KHẨU')."""
    return para == para.upper() and any(ch.isalpha() for ch in para) and not para[0].isdigit()


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode về NFC và khoảng trắng, giữ ranh giới dòng.

    - NFC: tiếng Việt có thể được mã hóa dựng sẵn hoặc tổ hợp (dấu rời);
      hai dạng nhìn giống nhau nhưng khác byte, làm hỏng so khớp evidence.
    - Thay NBSP/zero-width bằng khoảng trắng thường; gộp khoảng trắng trong dòng.
    - Giữ ``\\n`` vì chunking B dựa vào đầu dòng ``1.`` / ``a)``.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def read_docx_paragraphs(path: Path) -> list[str]:
    """Đọc paragraph của file .docx (bảng chữ ký và footnote không lấy)."""
    from docx import Document  # python-docx

    doc = Document(str(path))
    return [p.text for p in doc.paragraphs if p.text and p.text.strip()]


def read_html_paragraphs(path: Path) -> list[str]:
    """Đọc thẻ <p> trong khối toàn văn ``div.cldivContentDocVn``.

    Khoanh vùng khối toàn văn trước để không kéo theo menu/văn bản liên quan.
    Báo lỗi nếu không tìm thấy khối, không âm thầm đọc cả trang.
    """
    from bs4 import BeautifulSoup

    html = Path(path).read_text(encoding="utf-8", errors="strict")
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.cldivContentDocVn") or soup.select_one("#divContentDoc")
    if content is None:
        raise ValueError(f"{path}: không tìm thấy khối toàn văn div.cldivContentDocVn")
    # Mã HTML xuống dòng cứng bên trong một <p> (ví dụ "Điều 5. Hồ sơ, trình tự tự công\nbố")
    # nên mỗi <p> được gộp về một dòng; ranh giới đoạn = ranh giới thẻ <p>.
    paras = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in content.find_all("p")]
    # get_text(" ") chèn khoảng trắng ở ranh giới thẻ inline → "Nghị định này ." ; gỡ khoảng
    # trắng thừa trước dấu câu. (Chỉ số trên như "cm<sup>2</sup>" vẫn thành "cm 2" — ghi hạn chế.)
    paras = [re.sub(r"\s+([.,;:)])", r"\1", t) for t in paras]
    return [t for t in paras if t]


def split_into_articles(paragraphs: list[str], doc: dict[str, Any],
                        log: dict[str, Any]) -> list[DocumentPage]:
    """Duyệt một lượt các paragraph, mở đơn vị mới ở mỗi dòng ``Điều N.``.

    Phần trước Điều 1 (quốc hiệu, căn cứ) bị bỏ qua; dừng ở ``Nơi nhận:``
    (khối chữ ký, phụ lục biểu mẫu phía sau không thuộc phạm vi corpus).
    Tiêu đề Chương/Mục (kể cả tiêu đề nhiều dòng viết HOA) được tách thành
    metadata, không nối vào thân Điều đứng trước.
    """
    pages: list[DocumentPage] = []
    current: dict[str, Any] | None = None
    chuong_so, chuong_ten, cho_ten_chuong = "", "", False
    muc_so, muc_ten, heading_mode = "", "", None   # heading_mode: None | "chuong" | "muc"
    heading_lines = 0
    skipped_preamble = 0
    stopped_at = None

    def close() -> None:
        if current is None:
            return
        body = normalize_text("\n".join(current["lines"]))
        number = current["so_dieu"]
        meta = {
            "unit_id": f"{doc['document_id']}:D{number}",
            "so_hieu": doc["so_hieu"],
            "loai_van_ban": doc["loai_van_ban"],
            "title": doc["title"],
            "dieu": number,
            "ten_dieu": current["ten_dieu"],
            "chuong": current["chuong"],
            "ten_chuong": current["ten_chuong"],
            "muc": current["muc"],
            "ten_muc": current["ten_muc"],
        }
        heading = f"Điều {number}. {current['ten_dieu']}".strip()
        if not body:
            log["empty_articles"].append({"unit_id": meta["unit_id"], "heading": heading})
            return
        # Dòng tiêu đề là một phần của văn bản nguồn (tên Điều thường mang phạm vi/điều kiện,
        # ví dụ "Các trường hợp được miễn ... (trừ các trường hợp có cảnh báo ...)"), nên được giữ
        # ở đầu text của đơn vị; đồng thời vẫn có trong metadata ``ten_dieu``.
        pages.append(DocumentPage(document_id=doc["document_id"], source=doc["source_path"],
                                  page=number, text=f"{heading}\n{body}", metadata=meta))

    for raw in paragraphs:
        para = normalize_text(raw)
        if not para:
            continue
        if PATTERN_KET_THUC.match(para):
            stopped_at = para[:60]
            break
        if cho_ten_chuong:
            chuong_ten, cho_ten_chuong, heading_mode = para, False, "chuong"
            heading_lines += 1
            continue
        m_ch = PATTERN_CHUONG.match(para)
        if m_ch:
            chuong_so, cho_ten_chuong = m_ch.group(1), True
            muc_so, muc_ten = "", ""          # Mục đánh số lại theo từng Chương
            heading_lines += 1
            continue
        m_muc = PATTERN_MUC.match(para)
        if m_muc:
            muc_so, muc_ten, heading_mode = m_muc.group(1), "", "muc"
            heading_lines += 1
            continue
        if heading_mode and is_heading_line(para):
            if heading_mode == "chuong":
                chuong_ten = f"{chuong_ten} {para}".strip()
            else:
                muc_ten = f"{muc_ten} {para}".strip()
            heading_lines += 1
            continue
        heading_mode = None
        m_d = PATTERN_DIEU.match(para)
        if m_d:
            close()
            current = {"so_dieu": int(m_d.group(1)), "ten_dieu": m_d.group(2).strip(),
                       "chuong": chuong_so, "ten_chuong": chuong_ten, "muc": muc_so,
                       "ten_muc": muc_ten, "lines": []}
            continue
        if current is None:
            skipped_preamble += 1
        else:
            current["lines"].append(para)
    close()

    numbers = [p.page for p in pages] + [int(e["unit_id"].split(":D")[1]) for e in log["empty_articles"]
                                         if e["unit_id"].startswith(doc["document_id"] + ":")]
    dupes = sorted({n for n in numbers if numbers.count(n) > 1})
    expected = set(range(1, max(numbers) + 1)) if numbers else set()
    log["documents"][doc["document_id"]] = {
        "paragraphs_read": len(paragraphs),
        "preamble_paragraphs_skipped": skipped_preamble,
        "heading_lines_moved_to_metadata": heading_lines,
        "stopped_at": stopped_at,
        "articles_with_content": len(pages),
        "max_article_number": max(numbers) if numbers else 0,
        "missing_article_numbers": sorted(expected - set(numbers)),
        "duplicate_article_numbers": dupes,
        "content_chars": sum(len(p.text) for p in pages),
    }
    if dupes:
        log["errors"].append(f"{doc['document_id']}: số Điều bị trùng {dupes}")
    if expected - set(numbers):
        log["errors"].append(f"{doc['document_id']}: thiếu Điều {sorted(expected - set(numbers))}")
    return pages


def load_document(doc: dict[str, Any], repo_root: Path, log: dict[str, Any]) -> list[DocumentPage]:
    """Đọc một văn bản theo khai báo trong manifest (đuôi .docx hoặc .html)."""
    path = repo_root / doc["source_path"]
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file nguồn {path}")
    if path.suffix == ".docx":
        paragraphs = read_docx_paragraphs(path)
    elif path.suffix in (".html", ".htm"):
        paragraphs = read_html_paragraphs(path)
    else:
        raise ValueError(f"Định dạng chưa hỗ trợ: {path.suffix}")
    return split_into_articles(paragraphs, doc, log)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Corpus này không dùng PDF (bản PDF công khai là bản scan — xem báo cáo)."""
    raise NotImplementedError("Corpus dùng DOCX/HTML; PDF không nằm trong pipeline này")
