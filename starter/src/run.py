"""Pipeline Day 06. Chạy từ thư mục ``starter/``:

    python -m src.run verify-benchmark   # kiểm chứng evidence + chốt hash bộ câu hỏi
    python -m src.run ingest             # đọc, chuẩn hóa, chunk A/B, thống kê
    python -m src.run index              # embed + upsert pgvector (+ kiểm tra nạp lại)
    python -m src.run evaluate           # top-3 cho A/B/C, Hit@1/Hit@3, trace, file chấm tay
    python -m src.run summarize-manual   # tổng hợp nhãn chấm tay đã điền
    python -m src.run all                # bốn bước đầu
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import by_paragraph, fixed_size
from .config import COHERENCE_POSITIONS, HYPOTHESES, SELECTION_RULE, STRATEGIES, TOP_K
from .evaluation import contains_evidence, evaluate_query, evidence_parts, gold_items, summarize
from .ingestion import load_document
from .models import Chunk, DocumentPage

STARTER = Path(__file__).resolve().parents[1]
ROOT = STARTER.parent
DATA, BENCH, RESULTS = ROOT / "data", ROOT / "benchmark", ROOT / "results"
PROCESSED = DATA / "processed"
QUESTIONS = BENCH / "questions.jsonl"
FREEZE = BENCH / "freeze.json"
MANUAL = RESULTS / "manual"


# ---------- tiện ích ----------
def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(STARTER / ".env")
    except ImportError:
        pass


def model_name() -> str:
    from .embeddings import DEFAULT_MODEL
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL


def load_pages(log: dict[str, Any]) -> list[DocumentPage]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    pages: list[DocumentPage] = []
    for doc in manifest["documents"]:
        pages.extend(load_document(doc, ROOT, log))
    return pages


def pages_from_disk() -> list[DocumentPage]:
    return [DocumentPage(**r) for r in read_jsonl(PROCESSED / "pages.jsonl")]


def chunks_from_disk(name: str) -> list[Chunk]:
    return [Chunk(**r) for r in read_jsonl(PROCESSED / f"chunks_{name}.jsonl")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- 1. kiểm chứng benchmark ----------
def verify_benchmark(freeze: bool = True, refreeze_reason: str | None = None) -> dict[str, Any]:
    log = {"documents": {}, "errors": [], "empty_articles": []}
    units = {p.metadata["unit_id"]: p for p in load_pages(log)}
    qs = read_jsonl(QUESTIONS)
    problems, rows = [], []
    required = ["id", "split", "question", "expected_answer", "gold_source_id", "gold_page",
                "evidence_quote"]
    for q in qs:
        miss = [k for k in required if q.get(k) in (None, "")]
        if miss:
            problems.append(f"{q.get('id')}: thiếu trường {miss}")
            continue
        for g in gold_items(q):
            uid = f"{g['gold_source_id']}:D{g['gold_page']}"
            ok = uid in units and contains_evidence(units[uid].text, g)
            rows.append({"id": q["id"], "unit_id": uid, "evidence_found": ok,
                         "evidence_parts": len(evidence_parts(g)),
                         "evidence_chars": sum(len(x) for x in evidence_parts(g))})
            if not ok:
                problems.append(f"{q['id']}: evidence không nằm trong {uid}")
        f = q.get("filter")
        if f:
            if set(f) & {"document_id", "so_hieu"}:
                problems.append(f"{q['id']}: filter dùng trường định danh nguồn — không cho phép")
            for k, v in f.items():
                if str(v) not in {str(p.metadata.get(k)) for p in units.values()}:
                    problems.append(f"{q['id']}: giá trị filter {k}={v} không có trong corpus")
    splits = {s: sum(q["split"] == s for q in qs) for s in ("dev", "test")}
    filt = {s: sum(bool(q.get("filter")) and q["split"] == s for q in qs) for s in ("dev", "test")}
    if splits["dev"] < 15 or splits["test"] < 5:
        problems.append(f"split chưa đủ 15 dev / 5 test: {splits}")
    if sum(filt.values()) < 3 or filt["test"] < 1:
        problems.append(f"cần ≥3 câu filter, ≥1 ở test: {filt}")
    if len({q['id'] for q in qs}) != len(qs):
        problems.append("id câu hỏi bị trùng")
    report = {"checked_at": now(), "n_questions": len(qs), "splits": splits,
              "filter_questions": filt, "evidence_checks": rows, "problems": problems,
              "questions_sha256": sha256(QUESTIONS)}
    write_json(RESULTS / "benchmark_verification.json", report)
    print(f"{len(qs)} câu {splits}, filter {filt}; vấn đề: {len(problems)}")
    for p in problems:
        print("  -", p)
    if problems:
        sys.exit(1)
    if freeze:
        if FREEZE.exists():
            old = json.loads(FREEZE.read_text(encoding="utf-8"))
            if old["questions_sha256"] != report["questions_sha256"]:
                if not refreeze_reason:
                    sys.exit("questions.jsonl đã khác bản đã chốt (benchmark/freeze.json). Nếu sửa có "
                             "chủ đích: chạy lại với --refreeze \"<lý do>\" (lý do được lưu vào lịch sử).")
                history = old.get("history", []) + [{
                    "frozen_at": old["frozen_at"], "questions_sha256": old["questions_sha256"],
                    "replaced_at": now(), "reason": refreeze_reason}]
                write_json(FREEZE, {"frozen_at": now(), "questions_sha256": report["questions_sha256"],
                                    "note": old.get("note", ""), "history": history})
                print("Đã chốt lại benchmark, lý do lưu trong freeze.json → history")
            else:
                print("Hash trùng bản đã chốt", old["frozen_at"])
        else:
            write_json(FREEZE, {"frozen_at": now(), "questions_sha256": report["questions_sha256"],
                                "note": "Chốt bộ câu hỏi + split trước khi chạy retrieval."})
            print("Đã chốt benchmark → benchmark/freeze.json")
    return report


# ---------- 2. ingest + chunk ----------
def ingest() -> None:
    from .embeddings import TokenCounter

    t0 = time.perf_counter()
    log = {"run_at": now(), "documents": {}, "errors": [], "empty_articles": []}
    pages = load_pages(log)
    t_read = time.perf_counter() - t0
    write_jsonl(PROCESSED / "pages.jsonl", [asdict(p) for p in pages])
    with (DATA / "manifest_units.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "document_id", "page_dieu", "ten_dieu", "chuong", "chars", "counted"])
        for p in pages:
            w.writerow([p.metadata["unit_id"], p.document_id, p.page, p.metadata["ten_dieu"],
                        p.metadata["chuong"], len(p.text), "yes"])
        for e in log["empty_articles"]:
            doc_id, num = e["unit_id"].split(":D")
            w.writerow([e["unit_id"], doc_id, num, e["heading"], "", 0, "no (không có nội dung)"])
    log["content_units"] = len(pages)
    log["content_chars"] = sum(len(p.text) for p in pages)
    log["units_of_1500_chars"] = log["content_chars"] // 1500
    if len(pages) < 30:
        log["errors"].append(f"Chỉ có {len(pages)} đơn vị nội dung (<30)")

    counter = TokenCounter(model_name())
    t1 = time.perf_counter()
    chunks = {"fixed": [c for p in pages for c in fixed_size(p, STRATEGIES["A"]["chunk_size"],
                                                              STRATEGIES["A"]["overlap"])]}
    t_fixed = time.perf_counter() - t1
    t1 = time.perf_counter()
    chunks["para"] = [c for p in pages for c in by_paragraph(p, STRATEGIES["B"]["chunk_size"], counter)]
    t_para = time.perf_counter() - t1
    stats = []
    for name, cs in chunks.items():
        write_jsonl(PROCESSED / f"chunks_{name}.jsonl", [asdict(c) for c in cs])
        chars = [len(c.text) for c in cs]
        toks = [counter(c.embedding_text) for c in cs]
        stats.append({"index": name, "strategies": "A" if name == "fixed" else "B,C",
                      "chunks": len(cs), "chars_total": sum(chars), "chars_min": min(chars),
                      "chars_median": statistics.median(chars), "chars_mean": round(statistics.mean(chars), 1),
                      "chars_max": max(chars), "tokens_total": sum(toks),
                      "tokens_median": statistics.median(toks), "tokens_max": max(toks),
                      "over_512_tokens_truncated": sum(t > 512 for t in toks),
                      "oversize_pieces": sum(bool(c.metadata.get("oversize")) for c in cs),
                      "chunk_seconds": round(t_fixed if name == "fixed" else t_para, 3)})
    log["read_seconds"] = round(t_read, 3)
    write_json(RESULTS / "ingest_log.json", log)
    with (RESULTS / "chunk_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    print(f"{len(pages)} Điều, {log['content_chars']} ký tự; lỗi: {log['errors'] or 'không'}")
    for s in stats:
        print(f"  {s['index']}: {s['chunks']} chunk, token max {s['tokens_max']}, "
              f">512: {s['over_512_tokens_truncated']}")


# ---------- 3. index ----------
def _store(dimensions: int):
    from .store import PgVectorStore
    store = PgVectorStore(os.environ.get("DATABASE_URL", ""), dimensions)
    store.create_schema()
    return store


def index(force: bool = False) -> None:
    from .embeddings import E5Embedder
    from .store import content_hash

    emb = E5Embedder(model_name())
    store = _store(emb.dimensions)
    old_log = (json.loads((RESULTS / "index_log.json").read_text(encoding="utf-8"))
               if (RESULTS / "index_log.json").exists() else {"indexes": {}})
    doc_ids = sorted({p.document_id for p in pages_from_disk()})
    report = {"run_at": now(), "embedder": emb.info(), "store": store.versions(),
              "search": "exact (không index ANN), cosine distance <=>", "indexes": {}}
    for name in ("fixed", "para"):
        chunks = chunks_from_disk(name)
        passes = []
        for attempt in ("lần 1", "nạp lại cùng phiên bản"):
            before = store.count(name)
            existing = {} if (force and attempt == "lần 1") else store.existing_hashes(name)
            hashes = [content_hash(c, emb.model_name) for c in chunks]
            todo = [i for i, (c, h) in enumerate(zip(chunks, hashes)) if existing.get(c.chunk_id) != h]
            t0 = time.perf_counter()
            vectors = emb.embed_documents([chunks[i].embedding_text for i in todo]) if todo else []
            t_embed = time.perf_counter() - t0
            t0 = time.perf_counter()
            written = store.upsert(name, [chunks[i] for i in todo], vectors, [hashes[i] for i in todo])
            todo_set = set(todo)
            meta_updated = store.update_metadata(name, [c for i, c in enumerate(chunks) if i not in todo_set])
            deleted = store.delete_obsolete(name, doc_ids, [c.chunk_id for c in chunks])
            t_up = time.perf_counter() - t0
            after = store.count(name)
            passes.append({"pass": attempt, "rows_before": before, "embedded": len(todo),
                           "written": written, "metadata_updated": meta_updated,
                           "deleted_obsolete": deleted, "rows_after": after,
                           "embed_seconds": round(t_embed, 2), "upsert_seconds": round(t_up, 2)})
            print(f"{name} [{attempt}]: embed {len(todo)} ({t_embed:.1f}s), cập nhật metadata "
                  f"{meta_updated}, rows {before}→{after}")
        ok = passes[-1]["embedded"] == 0 and passes[-1]["rows_after"] == len(chunks)
        # thời gian embed toàn bộ index: chỉ cập nhật khi lần 1 thực sự embed hết chunk
        full = (passes[0]["embed_seconds"] if passes[0]["embedded"] == len(chunks)
                else old_log["indexes"].get(name, {}).get("full_embed_seconds"))
        report["indexes"][name] = {"chunks": len(chunks), "passes": passes, "reload_no_duplicate": ok,
                                   "full_embed_seconds": full}
        if not ok:
            report.setdefault("errors", []).append(f"{name}: kiểm tra nạp lại không đạt")
    write_json(RESULTS / "index_log.json", report)
    store.close()


# ---------- 4. evaluate ----------
def _coherence_positions(pages: list[DocumentPage]) -> list[tuple[str, int]]:
    """10 vị trí chọn trước khi có điểm: Điều cách đều trong thứ tự (document_id, số Điều),
    vị trí = ký tự giữa Điều."""
    ordered = sorted(pages, key=lambda p: (p.document_id, p.page))
    n = len(ordered)
    picks = [ordered[(2 * i + 1) * n // (2 * COHERENCE_POSITIONS)] for i in range(COHERENCE_POSITIONS)]
    return [(p.metadata["unit_id"], len(p.text) // 2) for p in picks]


LABEL_COLS = {"coherence_labels.csv": "label_0_2", "metadata_audit.csv": "metadata_correct_0_1",
              "grounding_labels.csv": "label_0_2"}


def _has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8-sig") as f:
        return any((r.get(LABEL_COLS[path.name]) or "").strip() for r in csv.DictReader(f))


def _chunk_text(h: dict[str, Any]) -> str:
    return h.get("embedding_text") or h["text"]


def check_manual_matches(traces: dict[str, list[dict]]) -> list[str]:
    """Nhãn chấm tay có thuộc đúng lần chạy hiện tại không.

    So cả ID lẫn **nội dung và các trường được chấm** (ID chỉ ghép từ Điều + số part,
    nên nội dung/metadata có thể đổi mà ID giữ nguyên).
    """
    problems = []
    by = {(s, t["id"]): t for s, rows in traces.items() for t in rows}

    def rows(name: str) -> list[dict[str, str]]:
        p = MANUAL / name
        return list(csv.DictReader(p.open(encoding="utf-8-sig"))) if p.exists() else []

    for r in rows("grounding_labels.csv"):
        t = by.get((r["strategy"], r["question_id"]))
        tag = f"grounding {r['strategy']} {r['question_id']}"
        if not t:
            problems.append(f"{tag}: không có trong trace")
            continue
        texts = [_chunk_text(h) for h in t["top_k"]] + [""] * 3
        expect = {"top3_chunk_ids": " | ".join(h["chunk_id"] for h in t["top_k"]),
                  "question": t["question"], "expected_answer": t["expected_answer"],
                  "top1_text": texts[0], "top2_text": texts[1], "top3_text": texts[2]}
        diff = [k for k, v in expect.items() if r.get(k) != v]
        if diff:
            problems.append(f"{tag}: khác trace ở {diff}")
    for r in rows("metadata_audit.csv"):
        t = by.get((r["strategy"], r["question_id"]))
        k = int(r["rank"]) - 1
        tag = f"metadata {r['strategy']} {r['question_id']} hạng {r['rank']}"
        if not t or k >= len(t["top_k"]):
            problems.append(f"{tag}: không có trong trace")
            continue
        h = t["top_k"][k]
        expect = {"chunk_id": h["chunk_id"], "document_id": h["document_id"], "source": h["source"],
                  "page_dieu": str(h["page"]), "ten_dieu": h["metadata"].get("ten_dieu"),
                  "loai_van_ban": h["metadata"].get("loai_van_ban"),
                  "filter": json.dumps(t["filter"], ensure_ascii=False) if t["filter"] else "",
                  "chunk_text": _chunk_text(h)}
        diff = [k2 for k2, v in expect.items() if r.get(k2) != v]
        if diff:
            problems.append(f"{tag}: khác trace ở {diff}")
    current = {c.chunk_id: c.embedding_text for n in ("fixed", "para") for c in chunks_from_disk(n)}
    for r in rows("coherence_labels.csv"):
        if current.get(r["chunk_id"]) != r["chunk_text"]:
            problems.append(f"coherence {r['position_id']} {r['strategy']}: chunk {r['chunk_id']} đã đổi")
    for r in rows("evidence_confirmations.csv"):
        t = by.get((r["strategy"], r["question_id"]))
        h = next((h for h in (t or {}).get("top_k", []) if h["chunk_id"] == r["chunk_id"]), None)
        if h is None or _chunk_text(h) != r["chunk_text"]:
            problems.append(f"xác nhận {r['strategy']} {r['question_id']} {r['chunk_id']}: chunk không còn "
                            "trong top-3 hoặc nội dung đã đổi — cần duyệt lại")
    return problems


def load_confirmations() -> set[tuple[str, str, str]]:
    """(question_id, chunk_id, nội dung chunk đã duyệt) — chỉ áp dụng khi nội dung còn y nguyên."""
    p = MANUAL / "evidence_confirmations.csv"
    if not p.exists():
        return set()
    with p.open(encoding="utf-8-sig") as f:
        return {(r["question_id"], r["chunk_id"], r["chunk_text"]) for r in csv.DictReader(f)
                if r["confirmed_0_1"].strip() == "1"}


def _write_manual_templates(traces: dict[str, list[dict]], pages: list[DocumentPage],
                            overwrite: bool = False) -> None:
    """Sinh file chấm tay (UTF-8 có BOM để Excel hiển thị đúng tiếng Việt).

    Không bao giờ ghi đè file đã có nhãn; ``overwrite`` chỉ dùng để làm lại file còn trống.
    """
    MANUAL.mkdir(parents=True, exist_ok=True)
    made = []
    for name in LABEL_COLS:
        if overwrite and _has_labels(MANUAL / name):
            sys.exit(f"{name} đã có nhãn — không ghi đè.")
    path = MANUAL / "coherence_labels.csv"
    if overwrite or not path.exists():
        idx_chunks = {"fixed": chunks_from_disk("fixed"), "para": chunks_from_disk("para")}
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["position_id", "unit_id", "char_offset", "strategy", "chunk_id",
                        "chunk_text", "label_0_2", "reason"])
            for k, (uid, off) in enumerate(_coherence_positions(pages), 1):
                for strat in ("A", "B", "C"):
                    cs = idx_chunks[STRATEGIES[strat]["index"]]
                    own = [c for c in cs if c.metadata["parent_id"] == uid]
                    # chunk đầu tiên phủ vị trí; nếu vị trí rơi vào dấu xuống dòng giữa hai part
                    # của B thì lấy part bắt đầu ngay sau đó
                    c = next((c for c in own if c.metadata["char_start"] <= off < c.metadata["char_end"]),
                             None) or min(own, key=lambda c: abs(c.metadata["char_start"] - off))
                    note = "C dùng chung chunk với B → nhãn như B" if strat == "C" else ""
                    w.writerow([f"P{k:02d}", uid, off, strat, c.chunk_id, c.embedding_text, "", note])
        made.append(path.name)
    path = MANUAL / "metadata_audit.csv"
    if overwrite or not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["strategy", "question_id", "rank", "chunk_id", "document_id", "source",
                        "page_dieu", "ten_dieu", "loai_van_ban", "filter", "filter_ok_auto",
                        "chunk_text", "metadata_correct_0_1", "note"])
            for strat, rows in traces.items():
                for t in rows:
                    if t["split"] != "test":
                        continue
                    for r, h in enumerate(t["top_k"], 1):
                        w.writerow([strat, t["id"], r, h["chunk_id"], h["document_id"], h["source"],
                                    h["page"], h["metadata"].get("ten_dieu"),
                                    h["metadata"].get("loai_van_ban"),
                                    json.dumps(t["filter"], ensure_ascii=False) if t["filter"] else "",
                                    h["filter_ok"], h.get("embedding_text") or h["text"], "", ""])
        made.append(path.name)
    path = MANUAL / "grounding_labels.csv"
    if overwrite or not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["strategy", "question_id", "question", "expected_answer", "top3_chunk_ids",
                        "top1_text", "top2_text", "top3_text", "auto_evidence_hit@3", "label_0_2",
                        "reason"])
            for strat, rows in traces.items():
                for t in rows:
                    if t["split"] == "test":
                        texts = [h.get("embedding_text") or h["text"] for h in t["top_k"]]
                        texts += [""] * (3 - len(texts))
                        w.writerow([strat, t["id"], t["question"], t["expected_answer"],
                                    " | ".join(h["chunk_id"] for h in t["top_k"]), *texts[:3],
                                    t["eval"]["hit@3"], "", ""])
        made.append(path.name)
    print("File chấm tay mới:", made or "không (đã có, không ghi đè)")


def evaluate() -> None:
    from .embeddings import E5Embedder

    if not FREEZE.exists() or json.loads(FREEZE.read_text(encoding="utf-8"))["questions_sha256"] != sha256(QUESTIONS):
        sys.exit("Benchmark chưa chốt hoặc đã bị sửa sau khi chốt — chạy verify-benchmark trước.")
    emb = E5Embedder(model_name())
    store = _store(emb.dimensions)
    qs = read_jsonl(QUESTIONS)
    confirmed = load_confirmations()
    t0 = time.perf_counter()
    qvecs = {q["id"]: emb.embed_query(q["question"]) for q in qs}
    t_q = time.perf_counter() - t0
    all_traces: dict[str, list[dict]] = {}
    timing = {}
    for strat, cfg in STRATEGIES.items():
        rows = []
        t0 = time.perf_counter()
        for q in qs:
            filters = q.get("filter") if cfg["use_filter"] else None
            hits = store.search(cfg["index"], qvecs[q["id"]], TOP_K, filters)
            ev = evaluate_query(q, hits, confirmed)
            rows.append({"strategy": strat, "id": q["id"], "split": q["split"], "type": q.get("type"),
                         "question": q["question"], "expected_answer": q["expected_answer"],
                         "gold": gold_items(q), "filter": q.get("filter"),
                         "filter_applied": filters, "eval": {k: v for k, v in ev.items() if k != "per_hit"},
                         "top_k": [{"rank": r, "chunk_id": h["chunk_id"], "score": round(float(h["score"]), 4),
                                    "document_id": h["document_id"], "source": h["source"],
                                    "page": h["page"], "metadata": h["metadata"], "text": h["text"],
                                    "embedding_text": h["embedding_text"],
                                    **ph} for r, (h, ph) in enumerate(zip(hits, ev["per_hit"]), 1)]})
        timing[strat] = round(time.perf_counter() - t0, 3)
        write_jsonl(RESULTS / "traces" / f"{strat}.jsonl", rows)
        all_traces[strat] = rows
    store.close()

    # kiểm tra giả thuyết phụ: câu không filter, C phải trùng B
    same = sum([h["chunk_id"] for h in b["top_k"]] == [h["chunk_id"] for h in c["top_k"]]
               for b, c in zip(all_traces["B"], all_traces["C"]) if not b["filter"])
    n_nofilter = sum(not q.get("filter") for q in qs)
    summary = summarize([t for rows in all_traces.values() for t in rows])
    chunk_stats = {r["index"]: r for r in csv.DictReader((RESULTS / "chunk_stats.csv").open(encoding="utf-8"))}
    index_log = json.loads((RESULTS / "index_log.json").read_text(encoding="utf-8"))
    metrics = {"run_at": now(), "embedder": emb.info(), "top_k": TOP_K,
               "selection_rule": SELECTION_RULE, "query_embed_seconds": round(t_q, 3),
               "search_seconds": timing, "summary": summary,
               "check_C_equals_B_without_filter": f"{same}/{n_nofilter}"}
    write_json(RESULTS / "metrics.json", metrics)

    lines = ["| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Hit@3 so khớp tự động | "
             "Filter tuân thủ | Chunk | Embed (s) | Chi phí |",
             "|---|---|---|---|---|---|---|---|---|"]
    for strat, cfg in STRATEGIES.items():
        cs = chunk_stats[cfg["index"]]
        emb_s = index_log["indexes"][cfg["index"]].get("full_embed_seconds")
        for split in ("dev", "test"):
            s = summary[f"{strat}|{split}"]
            lines.append(f"| {strat} | {split} | {s['hit@1']} | {s['evidence_hit@3']} | "
                         f"{s['evidence_hit@3_auto']} | {s['filter_compliance']} | "
                         f"{cs['chunks']} | {emb_s} | $0 |")
    (RESULTS / "metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"C trùng B ở câu không filter: {same}/{n_nofilter}")
    stale = check_manual_matches(all_traces) if MANUAL.exists() else []
    metrics["manual_labels_match_this_run"] = not stale
    write_json(RESULTS / "metrics.json", metrics)
    if stale:
        print("CẢNH BÁO: nhãn chấm tay trong results/manual/ không khớp lần chạy này "
              f"({len(stale)} dòng) — cần chấm lại, ví dụ: {stale[:3]}")
    elif MANUAL.exists():
        print("Nhãn chấm tay khớp với top-3/chunk của lần chạy này.")

    # experiment log
    with (ROOT / "artifacts" / "experiment_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "chunking", "chunk_size", "overlap", "metadata", "filter", "vectorizer",
                    "hypothesis", "dev_hit1", "dev_hit3", "test_hit1", "test_hit3", "notes"])
        for strat, cfg in STRATEGIES.items():
            d, t = summary[f"{strat}|dev"], summary[f"{strat}|test"]
            w.writerow([strat, cfg["chunking"], f"{cfg['chunk_size']} {cfg['size_unit']}", cfg["overlap"],
                        "document_id, source, page(Điều), loai_van_ban, so_hieu, chuong, ten_dieu",
                        "loai_van_ban khi câu hỏi có filter" if cfg["use_filter"] else "không",
                        f"{emb.model_name} ({emb.dimensions}d, local)", HYPOTHESES[strat],
                        d["hit@1"], d["evidence_hit@3"], t["hit@1"], t["evidence_hit@3"],
                        f"index={cfg['index']}; tiền tố tên Điều={'có' if cfg['prefix'] else 'không'}"])
    _write_manual_templates(all_traces, pages_from_disk())


# ---------- 5. tổng hợp nhãn chấm tay ----------
def summarize_manual() -> None:
    traces = {s: read_jsonl(RESULTS / "traces" / f"{s}.jsonl") for s in STRATEGIES}
    stale = check_manual_matches(traces)
    if stale:
        sys.exit(f"Nhãn chấm tay không khớp trace hiện tại ({len(stale)} dòng): {stale[:5]}")
    out: dict[str, Any] = {"run_at": now(), "matches_current_traces": True,
                           "equivalent_evidence_confirmed": sorted([q, c] for q, c, _ in load_confirmations())}

    def rows(name: str) -> list[dict[str, str]]:
        p = MANUAL / name
        return list(csv.DictReader(p.open(encoding="utf-8-sig"))) if p.exists() else []

    for name, col, key in (("coherence_labels.csv", "label_0_2", "coherence"),
                           ("grounding_labels.csv", "label_0_2", "grounding")):
        res = {}
        for r in rows(name):
            res.setdefault(r["strategy"], []).append(r[col].strip())
        out[key] = {}
        for strat, labels in res.items():
            filled = [int(x) for x in labels if x in ("0", "1", "2")]
            out[key][strat] = {"labelled": f"{len(filled)}/{len(labels)}",
                               "count_0_1_2": [filled.count(i) for i in (0, 1, 2)],
                               "mean_of_2": round(sum(filled) / len(filled), 2) if filled else None}
    meta = {}
    for r in rows("metadata_audit.csv"):
        m = meta.setdefault(r["strategy"], {"checked": 0, "correct": 0, "unlabelled": 0,
                                            "filter_q": set(), "filter_bad": set()})
        v = r["metadata_correct_0_1"].strip()
        if v in ("0", "1"):
            m["checked"] += 1
            m["correct"] += int(v)
        else:
            m["unlabelled"] += 1
        if r["filter"]:
            m["filter_q"].add(r["question_id"])
            if r["filter_ok_auto"] != "True":
                m["filter_bad"].add(r["question_id"])
    out["metadata"] = {s: {"correct": f"{m['correct']}/{m['checked']}", "unlabelled": m["unlabelled"],
                           "filter_compliance_test": (f"{len(m['filter_q']) - len(m['filter_bad'])}/{len(m['filter_q'])}"
                                                      if m["filter_q"] else "N/A")}
                       for s, m in meta.items()}
    write_json(RESULTS / "manual_summary.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 06 data ingestion pipeline")
    ap.add_argument("stage", choices=["verify-benchmark", "ingest", "index", "evaluate",
                                      "summarize-manual", "rebuild-manual", "all"])
    ap.add_argument("--force-reembed", action="store_true", help="embed lại toàn bộ ở lần 1")
    ap.add_argument("--no-freeze", action="store_true",
                    help="verify-benchmark chỉ kiểm tra, không tạo/so khớp benchmark/freeze.json")
    ap.add_argument("--refreeze", metavar="LÝ_DO",
                    help="chốt lại benchmark đã sửa có chủ đích; lý do được lưu vào freeze.json")
    args = ap.parse_args()
    load_env()
    if args.stage in ("verify-benchmark", "all"):
        verify_benchmark(freeze=not args.no_freeze, refreeze_reason=args.refreeze)
    if args.stage in ("ingest", "all"):
        ingest()
    if args.stage in ("index", "all"):
        index(force=args.force_reembed)
    if args.stage in ("evaluate", "all"):
        evaluate()
    if args.stage == "summarize-manual":
        summarize_manual()
    if args.stage == "rebuild-manual":   # làm lại file chấm tay còn trống từ trace đã lưu
        traces = {s: read_jsonl(RESULTS / "traces" / f"{s}.jsonl") for s in STRATEGIES}
        _write_manual_templates(traces, pages_from_disk(), overwrite=True)


if __name__ == "__main__":
    main()
