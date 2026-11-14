"""Hai cách chunk dùng cho ba cấu hình.

- A ``fixed_size``: cửa sổ ký tự cố định có overlap, cắt bất kể ranh giới câu;
  không thêm tiền tố. Đây là baseline.
- B/C ``by_paragraph``: theo cấu trúc văn bản pháp luật. Một Điều vừa giới hạn
  thì giữ nguyên; Điều quá dài được cắt đệ quy theo tầng Khoản (``1.``) rồi
  Điểm (``a)``), sau đó gom tham lam các mảnh liền kề cho tới sát giới hạn.
  Part đầu đã chứa sẵn dòng ``Điều N. <tên Điều>`` của nguồn; các part sau được
  thêm tiền tố đó vào ``embedding_text`` (contextual chunking).

Text của mỗi đơn vị Điều bắt đầu bằng dòng tiêu đề ``Điều N. <tên Điều>`` (là một
phần của văn bản nguồn), nên chunk đầu tiên của A cũng chứa tiêu đề.
  Kế thừa từ ``split_long_chunk`` của Project 4; chỉnh lại để đo độ dài bằng hàm
  truyền vào (token của model khi chạy thật) và để ``text`` luôn là chuỗi con
  nguyên văn của Điều (ghi được ``char_start``/``char_end``).

Mọi chunk đều mang ``document_id``, ``source``, ``page`` (số Điều) và metadata
lọc ``loai_van_ban`` sao chép từ đơn vị nguồn.
"""
from __future__ import annotations

import re
from typing import Callable

from .models import Chunk, DocumentPage

LengthFn = Callable[[str], int]
LEVEL_PATTERNS = [re.compile(r"\n(?=\d+\.\s)"),        # Khoản: "\n1. "
                  re.compile(r"\n(?=[a-zđ]\)\s)")]     # Điểm: "\na) "


def _make_chunk(page: DocumentPage, strategy: str, idx: int, text: str, start: int,
                embedding_text: str, n_parts: int) -> Chunk:
    meta = dict(page.metadata)
    meta.update({"strategy_family": strategy, "parent_id": page.metadata["unit_id"],
                 "part": idx + 1, "n_parts": n_parts,
                 "char_start": start, "char_end": start + len(text)})
    return Chunk(chunk_id=f"{page.metadata['unit_id']}#{strategy}{idx + 1}",
                 document_id=page.document_id, source=page.source, page=page.page,
                 text=text, metadata=meta, embedding_text=embedding_text)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Cửa sổ ``size`` ký tự, bước ``size - overlap``; không vượt ranh giới Điều."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("cần size > 0 và 0 <= overlap < size")
    text, spans, start = page.text, [], 0
    while True:
        end = min(start + size, len(text))
        spans.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return [_make_chunk(page, "fixed", i, text[s:e], s, text[s:e], len(spans))
            for i, (s, e) in enumerate(spans)]


def article_heading(page: DocumentPage) -> str:
    return f"Điều {page.page}. {page.metadata.get('ten_dieu', '')}".strip()


def by_paragraph(page: DocumentPage, max_size: int, length_fn: LengthFn = len,
                 add_prefix: bool = True) -> list[Chunk]:
    """Chunk theo Điều → Khoản → Điểm, đo độ dài *của embedding_text* bằng ``length_fn``.

    Một mảnh đơn lẻ vẫn vượt ``max_size`` sau khi hết tầng cắt được giữ nguyên
    (``metadata['oversize']=True``) để ghi log, không cắt giữa câu.
    """
    heading = article_heading(page)

    def with_context(content: str) -> str:
        """Part đã bắt đầu bằng tiêu đề Điều thì giữ nguyên; part khác thêm tiêu đề làm tiền tố."""
        if not add_prefix or content.startswith(heading):
            return content
        return f"{heading}\n{content}"

    def fits(content: str) -> bool:
        return length_fn(with_context(content)) <= max_size

    def split_and_group(content: str, level: int) -> list[str]:
        if fits(content) or level >= len(LEVEL_PATTERNS):
            return [content]
        parts = [p.strip() for p in LEVEL_PATTERNS[level].split(content) if p.strip()]
        if len(parts) <= 1:
            return split_and_group(content, level + 1)
        pieces: list[str] = []
        for part in parts:
            pieces.extend([part] if fits(part) else split_and_group(part, level + 1))
        groups: list[str] = []
        current: list[str] = []
        for piece in pieces:
            candidate = "\n".join(current + [piece])
            if current and not fits(candidate):
                groups.append("\n".join(current))
                current = [piece]
            else:
                current.append(piece)
        if current:
            groups.append("\n".join(current))
        return groups

    contents = split_and_group(page.text, 0)
    chunks, cursor = [], 0
    for i, content in enumerate(contents):
        start = page.text.find(content, cursor)
        if start < 0:  # không thể xảy ra nếu ghép bằng "\n" như nguồn; báo lỗi thay vì đoán
            raise RuntimeError(f"{page.metadata['unit_id']}: không định vị được part {i + 1}")
        cursor = start + len(content)
        ch = _make_chunk(page, "para", i, content, start, with_context(content), len(contents))
        ch.metadata["oversize"] = not fits(content)
        chunks.append(ch)
    return chunks
