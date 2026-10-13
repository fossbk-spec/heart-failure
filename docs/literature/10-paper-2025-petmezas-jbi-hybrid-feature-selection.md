# Paper Summary: Petmezas et al. (J Biomed Inform 2025)

**Tiêu đề**: Enhanced heart failure mortality prediction through model-independent hybrid feature selection and explainable machine learning
**Tác giả**: Georgios Petmezas, Vasileios E. Papageorgiou, Vassilios Vassilikos, … Nicos Maglaveras (Aristotle University of Thessaloniki; Northwestern University)
**Tạp chí**: *Journal of Biomedical Informatics* 163: 104800 (2025) — DOI [10.1016/j.jbi.2025.104800](https://doi.org/10.1016/j.jbi.2025.104800) — PMID 39956346

> Nguồn trích xuất: abstract PubMed. Cỡ mẫu và số liệu hiệu năng không có trong abstract — bắt buộc đối chiếu PDF trước khi đưa vào ma trận y văn.

---

## 1. Mục tiêu & Dữ liệu
- Dự đoán tử vong mọi nguyên nhân sau **1 năm** ở bệnh nhân suy tim từ dữ liệu **siêu âm tim** và một số đặc trưng nhân khẩu học.

## 2. Phương pháp
- Chọn đặc trưng lai, **độc lập mô hình**: Extra-Trees + độ đo tương quan phi tuyến; số đặc trưng tối ưu chọn qua đồ thị loss → **7 đặc trưng**.
- 7 mô hình ML trên bộ đầy đủ và bộ rút gọn; SHAP để diễn giải; so sánh với các kỹ thuật chọn đặc trưng phổ biến.

## 3. Kết quả cốt lõi
- Đa số mô hình giữ nguyên hoặc cải thiện hiệu năng dù giảm **80%** số đặc trưng.
- Bộ đặc trưng đề xuất vượt các phương pháp chọn đặc trưng thông thường (theo abstract).

## 4. Liên hệ với dự án
- Cùng hướng với Chicco & Jurman (2020): một tập nhỏ đặc trưng lâm sàng có thể đủ. Lưu ý: chọn đặc trưng phải nằm **bên trong** vòng CV để tránh rò rỉ.
