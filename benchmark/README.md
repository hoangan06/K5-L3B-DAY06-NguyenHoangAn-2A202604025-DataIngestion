# Benchmark

`questions.jsonl`: 20 câu, mỗi dòng gồm `id`, `split`, `type`, `question`, `expected_answer`, `gold_source_id`, `gold_page` (số Điều), `evidence_quote`, `filter` (hoặc `null`) và `alternatives` (gold thay thế cũng được chấp nhận).

Quy tắc evidence: đoạn evidence, đọc riêng, phải (a) xác định được phạm vi/chủ thể mà câu hỏi hỏi và (b) chứa đáp án cùng điều kiện/ngoại lệ đi kèm. Tên Điều chỉ được đưa vào evidence khi nó mang phạm vi, chủ thể hoặc ngoại lệ cần thiết mà thân Điều không có (q07, q09, q19); khi đó `evidence_quote` là danh sách các phần và chunk phải chứa tất cả các phần.

Phân chia (chốt trước khi chạy retrieval): 15 dev (q01–q15) / 5 test (q16–q20). Câu có filter: q08, q09, q10 (dev) và q16, q17 (test).

| Loại | Câu |
|---|---|
| truc_tiep | q01, q02, q05, q07, q12, q14, q20 |
| dien_dat_lai | q04, q13, q15, q19 |
| dieu_kien_ngoai_le | q03, q06, q11, q18 |
| metadata_filter | q08, q09, q10, q16, q17 |

Filter: `{"loai_van_ban": "luat" | "nghi_dinh"}` lấy từ chính câu hỏi ("Theo Luật…", "Theo Nghị định…"), tức phạm vi văn bản do người hỏi cung cấp trước khi truy xuất. Không lọc theo `document_id`/`so_hieu` (định danh gold). Với corpus hai văn bản, `loai_van_ban` trùng khớp một-một với văn bản, đây là hạn chế đã ghi trong báo cáo.

Kiểm chứng evidence: `python -m src.run verify-benchmark` kiểm tra mỗi `evidence_quote` (và mỗi gold thay thế) có nằm nguyên văn, sau chuẩn hóa NFC/chữ thường/khoảng trắng, trong Điều gold đã chuẩn hóa. Gold thay thế được tìm bằng cách dò cụm từ khóa trên toàn corpus: q05 (NĐ 15 Điều 19 cũng nêu mức 5%), q13 (NĐ 15 Điều 34 nêu cùng hai trường hợp truy xuất).

Nguy cơ rò rỉ từ dev sang test: test không dùng lại Điều nào của dev, nhưng cùng chủ đề rộng với dev ở mảng nhãn (q16, q18 ~ q07, q10) và kiểm tra nhập khẩu (q17, q19 ~ q05, q06).

Nguồn gốc câu hỏi: bản nháp do Claude soạn từ text đã chuẩn hóa; tôi duyệt và chỉnh trước khi chốt (`freeze.json`). Không dùng câu ngoài phạm vi corpus cho chỉ số bắt buộc.

Lịch sử sửa trước khi chốt (chưa chạy retrieval lần nào): khi duyệt, tôi phát hiện evidence của q19 không đủ thông tin (thiếu phạm vi "được miễn kiểm tra nhà nước" và ngoại lệ "trừ trường hợp có cảnh báo", vốn nằm ở tên Điều 13). Tôi rà lại cả 20 câu theo quy tắc evidence ở trên và sửa q07, q09, q19 (thêm phần tên Điều/phạm vi) và q13, q15 (mở rộng quote cho đủ chủ thể/hệ quả). Đồng thời sửa pipeline: giữ dòng tiêu đề Điều trong text của đơn vị. Việc sửa diễn ra sau khi kiểm tra chunk A/B có thể chứa evidence hay không (không embedding, không retrieval), được thúc đẩy bởi tiêu chí đủ bằng chứng và áp dụng đồng loạt cho cả 20 câu.

Đính chính sau khi chạy `evaluate` (không đổi cấu hình nào): q04: thu hẹp `expected_answer` còn ý "không thuộc diện cấp Giấy chứng nhận", vì evidence không hỗ trợ ý "vẫn phải tuân thủ điều kiện tương ứng" (không ảnh hưởng chỉ số; chốt lại bằng `--refreeze`, lý do lưu trong `freeze.json`). Bằng chứng tương đương (GUIDE §4.1): xét mọi trường hợp có chunk đúng Điều gold trong top-3 nhưng không khớp nguyên văn quote, có 6 trường hợp, đều thuộc A; chỉ q20 được xác nhận (chunk đơn lẻ chứa đủ đáp án, chỉ thiếu 3 chữ cuối của quote), các trường hợp cần ghép hai chunk như q17 không tính (`results/manual/evidence_confirmations.csv`).
