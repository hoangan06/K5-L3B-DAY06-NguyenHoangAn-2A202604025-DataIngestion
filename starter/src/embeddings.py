"""Embedding model thật chạy local: ``intfloat/multilingual-e5-base`` (768 chiều).

Lý do chọn (giữ từ Project 4): đa ngữ, hỗ trợ tiếng Việt, context 512 token,
chạy local nên chi phí 0 và không gửi tài liệu tới bên thứ ba.
Cách xử lý theo model card của E5: document thêm tiền tố ``"passage: "``, query
thêm ``"query: "``; vector được chuẩn hóa L2 nên cosine = tích vô hướng.
Không có fallback: lỗi tải model/thư viện sẽ dừng chương trình.
"""
from __future__ import annotations

from typing import Any, Protocol

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DOC_PREFIX, QUERY_PREFIX = "passage: ", "query: "


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class TokenCounter:
    """Đếm token đúng như lúc embed document (có ``passage: `` và token đặc biệt).

    Chỉ tải tokenizer (nhẹ), dùng ở bước chunking B để giới hạn theo token thật.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text: str) -> int:
        return len(self.tokenizer(DOC_PREFIX + text, add_special_tokens=True,
                                  truncation=False)["input_ids"])


class E5Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 32) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        # sentence-transformers 5.x có thể đổi tên hàm lấy số chiều; thử cả hai, không đoán số chiều
        get_dim = (getattr(self.model, "get_embedding_dimension", None)
                   or getattr(self.model, "get_sentence_embedding_dimension"))
        self.dimensions = int(get_dim())
        self.max_seq_length = int(self.model.max_seq_length)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=len(texts) > 64, convert_to_numpy=True)
        if vectors.shape != (len(texts), self.dimensions):
            raise RuntimeError(f"embedding shape bất thường: {vectors.shape}")
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([DOC_PREFIX + t for t in texts])

    def embed_query(self, question: str) -> list[float]:
        return self._encode([QUERY_PREFIX + question])[0]

    def info(self) -> dict[str, Any]:
        import sentence_transformers
        import torch
        import transformers

        return {"provider": "local (sentence-transformers)", "model": self.model_name,
                "dimensions": self.dimensions, "max_seq_length": self.max_seq_length,
                "device": self.device, "normalize": True,
                "document_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX,
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__, "torch": torch.__version__,
                "cost_usd": 0.0}


# Tên giữ theo starter
MyEmbedder = E5Embedder
