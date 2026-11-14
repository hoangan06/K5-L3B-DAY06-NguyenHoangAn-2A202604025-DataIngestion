"""Lưu và tìm vector bằng PostgreSQL + pgvector (Docker trong ``starter/``).

- Một bảng ``day6_chunks`` cho mọi cấu hình, khóa chính ``(index_name, chunk_id)``.
  A dùng index ``fixed``; B và C dùng chung index ``para`` (C chỉ khác ở filter).
- Tìm kiếm **exact** (không tạo index ANN), metric cosine: ``embedding <=> q``;
  ``score = 1 - cosine_distance``.
- Nạp lại idempotent: ``ON CONFLICT DO UPDATE``; chunk của văn bản đang nạp mà
  không còn trong bản mới bị xóa (xử lý tài liệu đổi). ``content_hash`` chỉ dùng
  để bỏ qua bước *embed* khi đầu vào embedding không đổi; metadata và text luôn
  được ghi lại (``update_metadata``) nên không bị cũ.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import Chunk

TABLE = "day6_chunks"
FILTERABLE = {"loai_van_ban", "so_hieu", "document_id", "chuong"}


def content_hash(chunk: Chunk, model_name: str) -> str:
    payload = json.dumps([model_name, chunk.embedding_text, chunk.document_id, chunk.page,
                          chunk.metadata.get("loai_van_ban")], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        import psycopg

        if not database_url:
            raise ValueError("DATABASE_URL trống — copy starter/.env.example thành starter/.env")
        if dimensions <= 0:
            raise ValueError("dimensions phải > 0")
        self.dimensions = dimensions
        self.conn = psycopg.connect(database_url, autocommit=True)

    def versions(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT current_setting('server_version'), "
                        "(SELECT extversion FROM pg_extension WHERE extname='vector')")
            pg, vec = cur.fetchone()
        return {"postgres": pg, "pgvector": vec or "chưa bật"}

    def create_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    index_name    TEXT NOT NULL,
                    chunk_id      TEXT NOT NULL,
                    document_id   TEXT NOT NULL,
                    source        TEXT NOT NULL,
                    page          INTEGER NOT NULL,
                    loai_van_ban  TEXT NOT NULL,
                    so_hieu       TEXT NOT NULL,
                    chuong        TEXT,
                    text          TEXT NOT NULL,
                    embedding_text TEXT NOT NULL,
                    metadata      JSONB NOT NULL,
                    content_hash  TEXT NOT NULL,
                    embedding     vector({self.dimensions}) NOT NULL,
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (index_name, chunk_id))""")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_filter_idx "
                        f"ON {TABLE} (index_name, loai_van_ban)")
            cur.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                        "WHERE attrelid = %s::regclass AND attname = 'embedding'", (TABLE,))
            col_type = cur.fetchone()[0]
        if col_type != f"vector({self.dimensions})":
            raise RuntimeError(f"Bảng có cột {col_type}, model cần vector({self.dimensions})")

    def existing_hashes(self, index_name: str) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT chunk_id, content_hash FROM {TABLE} WHERE index_name=%s",
                        (index_name,))
            return dict(cur.fetchall())

    def count(self, index_name: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*), count(DISTINCT chunk_id) FROM {TABLE} "
                        f"WHERE index_name=%s", (index_name,))
            n, distinct = cur.fetchone()
        if n != distinct:
            raise RuntimeError(f"{index_name}: phát hiện chunk_id trùng ({n} dòng, {distinct} id)")
        return n

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]],
               hashes: list[str]) -> int:
        """Ghi các chunk đã embed (``strategy`` = tên index). Trả về số dòng đã ghi."""
        if not (len(chunks) == len(vectors) == len(hashes)):
            raise ValueError("số chunk, vector và hash không khớp")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"vector {len(v)} chiều, store cần {self.dimensions}")
        rows = [(strategy, c.chunk_id, c.document_id, c.source, c.page,
                 c.metadata["loai_van_ban"], c.metadata["so_hieu"], c.metadata.get("chuong"),
                 c.text, c.embedding_text, json.dumps(c.metadata, ensure_ascii=False), h, _vec(v))
                for c, v, h in zip(chunks, vectors, hashes)]
        with self.conn.cursor() as cur:
            cur.executemany(f"""
                INSERT INTO {TABLE} (index_name, chunk_id, document_id, source, page, loai_van_ban,
                    so_hieu, chuong, text, embedding_text, metadata, content_hash, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::vector)
                ON CONFLICT (index_name, chunk_id) DO UPDATE SET
                    document_id=EXCLUDED.document_id, source=EXCLUDED.source, page=EXCLUDED.page,
                    loai_van_ban=EXCLUDED.loai_van_ban, so_hieu=EXCLUDED.so_hieu,
                    chuong=EXCLUDED.chuong, text=EXCLUDED.text,
                    embedding_text=EXCLUDED.embedding_text, metadata=EXCLUDED.metadata,
                    content_hash=EXCLUDED.content_hash, embedding=EXCLUDED.embedding,
                    updated_at=now()""", rows)
        return len(rows)

    def update_metadata(self, index_name: str, chunks: list[Chunk]) -> int:
        """Ghi lại text/metadata cho chunk giữ nguyên vector (không embed lại)."""
        rows = [(c.document_id, c.source, c.page, c.metadata["loai_van_ban"], c.metadata["so_hieu"],
                 c.metadata.get("chuong"), c.text, c.embedding_text,
                 json.dumps(c.metadata, ensure_ascii=False), index_name, c.chunk_id) for c in chunks]
        with self.conn.cursor() as cur:
            cur.executemany(f"""
                UPDATE {TABLE} SET document_id=%s, source=%s, page=%s, loai_van_ban=%s, so_hieu=%s,
                    chuong=%s, text=%s, embedding_text=%s, metadata=%s::jsonb, updated_at=now()
                WHERE index_name=%s AND chunk_id=%s""", rows)
        return len(rows)

    def delete_obsolete(self, index_name: str, document_ids: list[str],
                        keep_chunk_ids: list[str]) -> int:
        """Xóa chunk cũ của các văn bản vừa nạp lại nhưng không còn trong bản mới."""
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE index_name=%s AND document_id = ANY(%s) "
                        f"AND NOT (chunk_id = ANY(%s))", (index_name, document_ids, keep_chunk_ids))
            return cur.rowcount

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Lọc metadata bằng SQL (tham số hóa, chỉ cột cho phép) rồi xếp theo cosine."""
        where, params = ["index_name = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key not in FILTERABLE:
                raise ValueError(f"không cho phép lọc theo '{key}'")
            where.append(f"{key} = %s")
            params.append(value)
        qv = _vec(query_vector)
        sql = (f"SELECT chunk_id, document_id, source, page, text, embedding_text, metadata, "
               f"1 - (embedding <=> %s::vector) AS score FROM {TABLE} "
               f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector LIMIT %s")
        with self.conn.cursor() as cur:
            cur.execute(sql, [qv, *params, qv, top_k])
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
