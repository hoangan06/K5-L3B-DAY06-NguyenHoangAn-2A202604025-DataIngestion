"""Chấm truy xuất tự động (Hit@1, Evidence Hit@3) và giữ trace từng câu.

Định nghĩa (GUIDE §4.1): một kết quả là *hit* khi chunk thuộc đúng nguồn + Điều
gold **và** chứa đủ ``evidence_quote`` (so khớp sau chuẩn hóa NFC, chữ thường,
khoảng trắng). ``evidence_quote`` là một chuỗi hoặc danh sách chuỗi; dạng danh
sách dùng khi bằng chứng gồm các phần không liền nhau trong nguồn (ví dụ tên
Điều mang phạm vi/điều kiện + một mục trong danh sách) và **mọi phần** phải có
trong chunk. Nội dung chunk được so là ``embedding_text`` — đúng phần văn bản
chunk mang theo khi lưu và embed (B có thêm tiêu đề Điều ở part sau).
Chấp nhận mọi gold thay thế được ghi trong ``alternatives``, và chunk mà người chấm
xác nhận chứa *bằng chứng tương đương* (``results/manual/evidence_confirmations.csv``,
xét từng chunk đơn lẻ, không ghép chunk; xác nhận gắn với đúng nội dung chunk đã duyệt,
nội dung đổi thì xác nhận hết hiệu lực). Giữ cả kết quả so khớp tự động
(``hit@1_auto``, ``hit@3_auto``) để thấy chênh lệch.
Coherence, grounding và độ đúng metadata **không** suy ra ở đây — các nhãn đó
chấm tay trong ``results/manual/``.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def evidence_parts(gold: dict[str, Any]) -> list[str]:
    ev = gold["evidence_quote"]
    return [ev] if isinstance(ev, str) else list(ev)


def contains_evidence(text: str, gold: dict[str, Any]) -> bool:
    t = norm(text)
    return all(norm(part) in t for part in evidence_parts(gold))


def gold_items(question: dict[str, Any]) -> list[dict[str, Any]]:
    main = {"gold_source_id": question["gold_source_id"], "gold_page": question["gold_page"],
            "evidence_quote": question["evidence_quote"]}
    return [main] + list(question.get("alternatives") or [])


def judge_hit(hit: dict[str, Any], golds: list[dict[str, Any]]) -> dict[str, bool]:
    text = hit.get("embedding_text") or hit["text"]
    same_place = [g for g in golds
                  if hit["document_id"] == g["gold_source_id"] and int(hit["page"]) == int(g["gold_page"])]
    has_evidence = any(contains_evidence(text, g) for g in same_place)
    return {"source_page_match": bool(same_place), "evidence_match": has_evidence,
            "hit": bool(same_place) and has_evidence}


def complies(hit: dict[str, Any], filters: dict[str, Any] | None) -> bool | None:
    if not filters:
        return None
    meta = hit.get("metadata") or {}
    return all(str(meta.get(k, hit.get(k))) == str(v) for k, v in filters.items())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   confirmed: set[tuple[str, str, str]] | None = None) -> dict[str, Any]:
    golds = gold_items(question)
    judged = []
    for h in hits:
        j = judge_hit(h, golds)
        j["hit_auto"] = j["hit"]
        j["confirmed_equivalent"] = (j["source_page_match"] and not j["hit"]
                                     and (question["id"], h["chunk_id"], h.get("embedding_text") or h["text"])
                                     in (confirmed or set()))
        j["hit"] = j["hit"] or j["confirmed_equivalent"]
        judged.append(j)
    filt = question.get("filter")
    compliance = [complies(h, filt) for h in hits]
    return {
        "hit@1": bool(judged and judged[0]["hit"]),
        "hit@3": any(j["hit"] for j in judged[:3]),
        "hit@1_auto": bool(judged and judged[0]["hit_auto"]),
        "hit@3_auto": any(j["hit_auto"] for j in judged[:3]),
        # chẩn đoán: đúng Điều trong top-3 nhưng không chunk nào chứa đủ evidence
        "source_page@3": any(j["source_page_match"] for j in judged[:3]),
        "evidence_split": any(j["source_page_match"] for j in judged[:3])
                          and not any(j["hit"] for j in judged[:3]),
        "filter_compliant": None if not filt else all(compliance[:3]),
        "per_hit": [{**j, "filter_ok": c} for j, c in zip(judged, compliance)],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Gộp theo (strategy, split); báo n/N, không chỉ phần trăm."""
    out: dict[str, Any] = {}
    for t in traces:
        key = f"{t['strategy']}|{t['split']}"
        s = out.setdefault(key, {"strategy": t["strategy"], "split": t["split"], "N": 0,
                                 "hit1": 0, "hit3": 0, "hit1_auto": 0, "hit3_auto": 0,
                                 "source_page3": 0, "evidence_split": 0,
                                 "filter_N": 0, "filter_ok": 0})
        ev = t["eval"]
        s["N"] += 1
        s["hit1"] += ev["hit@1"]
        s["hit3"] += ev["hit@3"]
        s["hit1_auto"] += ev["hit@1_auto"]
        s["hit3_auto"] += ev["hit@3_auto"]
        s["source_page3"] += ev["source_page@3"]
        s["evidence_split"] += ev["evidence_split"]
        if t.get("filter"):
            s["filter_N"] += 1
            s["filter_ok"] += bool(ev["filter_compliant"])
    for s in out.values():
        s["hit@1"] = f"{s['hit1']}/{s['N']}"
        s["evidence_hit@3"] = f"{s['hit3']}/{s['N']}"
        s["evidence_hit@3_auto"] = f"{s['hit3_auto']}/{s['N']}"
        s["filter_compliance"] = f"{s['filter_ok']}/{s['filter_N']}" if s["filter_N"] else "N/A"
    return out
