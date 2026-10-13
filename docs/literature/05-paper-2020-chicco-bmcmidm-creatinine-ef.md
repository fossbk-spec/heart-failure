# Paper Summary: Chicco & Jurman (BMC Med Inform Decis Mak 2020)

**Tiêu đề**: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone
**Tác giả**: Davide Chicco (Krembil Research Institute, Toronto), Giuseppe Jurman (Fondazione Bruno Kessler, Trento)
**Tạp chí**: *BMC Medical Informatics and Decision Making* 20:16 (2020) — DOI [10.1186/s12911-020-1023-5](https://doi.org/10.1186/s12911-020-1023-5) — PMID 32013925

> Nguồn trích xuất: abstract PubMed. Số liệu chi tiết (MCC, F1, ROC-AUC theo từng mô hình) cần đối chiếu PDF trong SSoT trước khi đưa vào ma trận y văn.

---

## 1. Mục tiêu & Dữ liệu
- Tái phân tích bộ 299 bệnh nhân suy tim (thu thập 2015, Ahmad et al. 2017) bằng nhiều bộ phân loại ML để (a) dự đoán sống còn và (b) xếp hạng yếu tố nguy cơ.

## 2. Phương pháp
- Nhiều classifier ML + xếp hạng đặc trưng bằng ML và bằng kiểm định thống kê sinh học truyền thống, so sánh hai cách xếp hạng.
- Xây dựng mô hình chỉ với **2 đặc trưng**: serum creatinine và ejection fraction.
- Phân tích bổ sung có **đưa tháng theo dõi (follow-up month) vào mô hình**.

## 3. Kết quả cốt lõi
- Cả hai cách xếp hạng đều chọn **serum creatinine** và **ejection fraction** là 2 đặc trưng quan trọng nhất.
- Mô hình 2 đặc trưng cho dự đoán chính xác hơn mô hình dùng toàn bộ đặc trưng (theo abstract).

## 4. Hạn chế & khoảng trống (đánh giá của dự án)
- Phân tích có thời gian theo dõi tiềm ẩn **rò rỉ thời gian** nếu dùng để dự báo tại thời điểm nhập viện.
- Cỡ mẫu 299, một trung tâm, không có kiểm chứng ngoài.
- Dự án tái lập kết luận chính: SHAP trên Random Forest xếp `serum_creatinine` và `ejection_fraction` ở vị trí 1–2 (notebook 03).
