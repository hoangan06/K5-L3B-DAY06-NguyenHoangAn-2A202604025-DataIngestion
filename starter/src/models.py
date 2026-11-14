"""Shared data contracts; keep traceability from chunk back to source.

Corpus là văn bản pháp luật (DOCX/HTML, không có số trang vật lý), nên đơn vị
truy vết ``page`` là **số Điều** trong văn bản (GUIDE cho phép "vị trí truy vết
tương đương"). ``metadata['unit_id']`` là ID duy nhất của đơn vị, ví dụ
``nd15-2018:D7``.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPage:
    document_id: str
    source: str
    page: int                     # số Điều
    text: str                     # "Điều N. <tên Điều>" + "\n" + nội dung Điều đã chuẩn hóa
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str                     # nội dung hiển thị / dùng để chấm evidence
    metadata: dict[str, Any] = field(default_factory=dict)
    # Văn bản đưa vào embedding model (B/C có thêm tiền tố tên Điều). Mặc định = text.
    embedding_text: str = ""

    def __post_init__(self) -> None:
        if not self.embedding_text:
            self.embedding_text = self.text
