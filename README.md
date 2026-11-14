# Day 06 · Data Ingestion Engineering · Nguyễn Hoàng Ân (2A202604025, L3B)

Biến hai văn bản pháp luật về an toàn thực phẩm thành kho vector có metadata, rồi so sánh ba chiến lược chuẩn bị dữ liệu (A/B/C) trên cùng một bộ 20 câu hỏi có gold evidence. Đề bài: [GUIDE](GUIDE.md) · [RUBRIC](RUBRIC.md) · [SUBMISSION](SUBMISSION.md).

| Thành phần | Lựa chọn |
|---|---|
| Corpus | Luật An toàn thực phẩm (VBHN 61/VBHN-VPQH, .docx) + Nghị định 15/2018/NĐ-CP (.html), 115 Điều có nội dung, 133.588 ký tự (kể cả dòng tên Điều); manifest: [data/manifest.json](data/manifest.json), danh sách đơn vị: `data/manifest_units.csv` |
| Đơn vị truy vết (`page`) | Số Điều (văn bản không có trang vật lý) |
| Embedding | `intfloat/multilingual-e5-base`, local, 768 chiều, tiền tố `passage:`/`query:`, chuẩn hóa L2; chi phí $0 |
| Vector store | PostgreSQL 16 + pgvector (Docker trong `starter/`), cosine, exact search |
| Trường lọc | `loai_van_ban` ∈ {`luat`, `nghi_dinh`} |
| A | Fixed-size 500 ký tự, overlap 50 trên text của từng Điều (dòng đầu là tiêu đề "Điều N. <tên>"), không thêm tiền tố, không filter |
| B | Theo cấu trúc Điều → Khoản → Điểm, ≤ 500 token e5; part sau part đầu được thêm tiêu đề Điều làm tiền tố; không filter |
| C | Giống B (dùng chung index) + filter `loai_van_ban` khi câu hỏi có filter |

Giả thuyết và tiêu chí chọn chiến lược được chốt trước khi chạy: [starter/src/config.py](starter/src/config.py).

## Setup (WSL / Linux, chạy từ thư mục repo)

Cần Docker Desktop (bật WSL integration), Python 3.10+.

```bash
cd starter
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # bỏ qua nếu đã có torch (GPU)
pip install -r requirements.txt
cp .env.example .env        # DATABASE_URL có sẵn; EMBEDDING_MODEL để trống = dùng e5-base mặc định; không cần API key
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py   # kỳ vọng: Setup OK
```

Lần đầu chạy, model e5-base (~1,1 GB) được tải về cache của Hugging Face.

## Chạy pipeline (từ `starter/`, đã activate `.venv`)

```bash
python -m src.run verify-benchmark   # kiểm chứng evidence nằm đúng Điều gold, kiểm tra split/filter, chốt hash → benchmark/freeze.json
python -m src.run ingest             # đọc + chuẩn hóa + tách Điều → data/processed/, data/manifest_units.csv, results/ingest_log.json, results/chunk_stats.csv
python -m src.run index              # embed + upsert pgvector cho index fixed (A) và para (B/C), nạp lại lần 2 để kiểm tra không trùng → results/index_log.json
python -m src.run evaluate           # top-3 cho A/B/C, Hit@1/Evidence Hit@3 dev/test → results/traces/, results/metrics.json, results/metrics_table.md, artifacts/experiment_log.csv, results/manual/*.csv
# chấm tay results/manual/*.csv (cột label_0_2 / metadata_correct_0_1, reason)
python -m src.run summarize-manual   # → results/manual_summary.json (báo lỗi nếu nhãn không khớp top-3, nội dung hoặc metadata của lần chạy cuối)
```

`evaluate` từ chối chạy nếu `benchmark/questions.jsonl` khác bản đã chốt; sửa có chủ đích thì chốt lại bằng `verify-benchmark --refreeze "<lý do>"` (lý do lưu trong `freeze.json`). Bằng chứng tương đương do người chấm xác nhận (GUIDE §4.1) nằm trong `results/manual/evidence_confirmations.csv`, gắn với đúng nội dung chunk đã duyệt (nội dung đổi thì hết hiệu lực), và được `evaluate` tính vào Hit; `metrics.json` giữ cả kết quả so khớp tự động. Dừng database và giữ dữ liệu: `docker compose -p day6_student_starter stop`.

## Cấu trúc

```text
data/raw/                 nguồn (.docx Công báo; .html chỉ gồm khối toàn văn)
data/manifest.json        nguồn, quyền, ngày truy cập, quy tắc đếm đơn vị
benchmark/                questions.jsonl (20 câu, 15 dev / 5 test), README, freeze.json
starter/src/              ingestion, chunking, embeddings, store, evaluation, config, run
results/                  log, thống kê chunk, trace, chỉ số, nhãn chấm tay
artifacts/experiment_log.csv
report/individual/2A202604025_NguyenHoangAn.md
```

## Ghi công

Các hàm đọc DOCX/HTML, tách Chương/Điều và cắt chunk theo Khoản/Điểm được tôi lấy lại từ Project 4 của mình (RAG văn bản pháp quy FMCG) và chỉnh sửa thêm: chuẩn hóa NFC, tách tiêu đề Mục/Chương nhiều dòng khỏi thân Điều, ghi vị trí ký tự, đo độ dài bằng tokenizer thật. Code và bản nháp benchmark được viết với sự hỗ trợ của Claude (Anthropic); tôi đã đọc, kiểm tra và chịu trách nhiệm về bài làm. Chi tiết trong báo cáo.
