"""Cấu hình ba chiến lược — chốt TRƯỚC khi chạy benchmark.

Giả thuyết (dự đoán có thể bác bỏ), viết trước khi xem kết quả:
- A→B: B giữ trọn Khoản/Điểm và có tiền tố tên Điều, nên Evidence Hit@3 của B
  ≥ A trên dev; A thua rõ nhất ở câu có evidence dài hoặc nằm gần chỗ cửa sổ
  500 ký tự cắt ngang. Nếu A ≥ B trên dev thì giả thuyết sai.
- B→C: filter ``loai_van_ban`` chỉ đổi kết quả ở câu có filter; với câu không
  filter, top-3 của C phải trùng hệt B (kiểm tra được). Trên câu có filter, C
  không được thấp hơn B về Hit@1/Hit@3, và đạt 100% tuân thủ filter.

Lưu ý thiết kế: A–B thay đổi hai yếu tố cùng lúc (ranh giới chunk và tiền tố
tên Điều); kết quả A–B là so sánh *trọn gói* chiến lược chunking. B–C chỉ thay
một yếu tố (filter).

Tiêu chí chọn chiến lược (chốt trước khi xem test): ưu tiên Evidence Hit@3 trên
dev; hòa thì Hit@1 dev; hòa tiếp thì chọn cấu hình ít chunk hơn / đơn giản hơn.
Test chỉ chạy một lần sau khi đã chọn, để báo cáo.
"""

STRATEGIES = {
    "A": {"index": "fixed", "chunking": "fixed_size", "chunk_size": 500, "overlap": 50,
          "size_unit": "ký tự", "prefix": False, "use_filter": False},
    "B": {"index": "para", "chunking": "by_paragraph (Điều→Khoản→Điểm)", "chunk_size": 500,
          "overlap": 0, "size_unit": "token e5 (gồm 'passage: ' và tiền tố)", "prefix": True,
          "use_filter": False},
    "C": {"index": "para", "chunking": "giống B (dùng chung index)", "chunk_size": 500,
          "overlap": 0, "size_unit": "token e5", "prefix": True, "use_filter": True},
}
HYPOTHESES = {
    "A": "Baseline: cửa sổ 500 ký tự cắt ngang Khoản → evidence dài dễ bị tách, Hit@3 thấp hơn B",
    "B": "Chunk theo cấu trúc + tiền tố tên Điều → Evidence Hit@3 dev >= A",
    "C": "Như B + filter loai_van_ban khi câu hỏi có filter → không thấp hơn B trên câu filter; câu không filter giữ nguyên top-3",
}
TOP_K = 3
SELECTION_RULE = "Evidence Hit@3 dev > Hit@1 dev > ít chunk hơn"
COHERENCE_POSITIONS = 10
