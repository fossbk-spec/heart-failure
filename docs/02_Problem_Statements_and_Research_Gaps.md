# Phát Biểu Bài Toán, Khoảng Trống Nghiên Cứu & Đóng Góp
## Dự Đoán Tử Vong Do Suy Tim Bằng Dữ Liệu Lớn & Học Máy

---

## 1. Phát Biểu Bài Toán (Problem Statements)

### Bài toán 1: Gánh nặng Tiên lượng Y tế & Tái Nhập Viện do Suy Tim
Suy tim ảnh hưởng đến hơn 64 triệu người trên toàn cầu, là nguyên nhân hàng đầu gây nhập viện ở người trên 65 tuổi với tỷ lệ tử vong trong vòng 5 năm lên tới 50%. Việc nhận diện sớm bệnh nhân có nguy cơ tử vong cao tại thời điểm nhập viện là yêu cầu cấp thiết để tối ưu hóa nguồn lực chăm sóc và cá thể hóa phác đồ điều trị.

### Bài toán 2: Hạn chế của Thang Điểm Lâm Sàng & Mô Hình Thống Kê Truyền Thống
Các thang điểm tiên lượng truyền thống (Seattle Heart Failure Model - SHFM, MAGGIC, GWTG-HF) và các mô hình hồi quy tuyến tính (Logistic Regression, Cox Proportional Hazards) thường chỉ đạt khả năng phân tách vừa phải (modest discrimination, C-index ~ 0.65 – 0.73). Các phương pháp này khó bắt được các mối quan hệ phi tuyến phức tạp và tương tác đa chiều giữa vô số chỉ số sinh hóa, sinh hiệu của bệnh nhân.

### Bài toán 3: Thách Thức Đặc Thù Của Dữ Liệu Y Tế Bảng (Tabular EHR Data)
Dữ liệu hồ sơ sức khỏe điện tử thực tế đối mặt với 3 thách thức lớn:
1. **Mất cân bằng lớp nghiêm trọng (Class Imbalance)**: Số ca tử vong chiếm tỷ lệ thấp ($32.1\%$ ở UCI, $<10\%$ ở PhysioNet Zigong). Nếu sử dụng các chỉ số như Accuracy, mô hình sẽ bị chệch về lớp đa số (sống sót).
2. **Dữ liệu khuyết & Nhiễu (Missing Values & Noise)**: Các xét nghiệm cận lâm sàng (như LVEF, CPK) thường bị khuyết nghiêm trọng ($68.4\%$ khuyết LVEF ở PhysioNet Zigong).
3. **Số lượng đặc trưng lớn nhưng chồng chéo (Curse of Dimensionality & Multicollinearity)**: Việc thu thập quá nhiều chỉ số gây tốn kém chi phí xét nghiệm và làm mô hình ML bị overfitting.

---

## 2. Khoảng Trống Nghiên Cứu (Research Gaps)

Thông qua phân tích tổng hợp từ Meta-Analysis của **Sun et al. (2022)**, **Olsen et al. (2020)** và các công trình thực nghiệm (**Chicco & Jurman 2020**, **Ishaq et al. 2021**), nghiên cứu này xác định 5 khoảng trống chính:

### Gap 1: Sự Thiếu Vượt Trội Nhất Quán của ML & Nguy Cơ Sai Lệch Cao (PROBAST Bias)
* *Thực trạng*: Các bài báo thực nghiệm thường tuyên bố ML vượt trội, nhưng Meta-Analysis trên 280 mô hình (Sun et al. 2022) chứng minh ML **không thể hiện sự vượt trội rõ rệt** so với Logistic Regression trong dự đoán tái nhập viện (C-index 0.660 vs 0.678).
* *Nguyên nhân*: 97% nghiên cứu ML vi phạm quy trình PROBAST: không thực hiện bước hiệu chuẩn mô hình (Calibration analysis), dung lượng mẫu nhỏ (Event per Variable < 10), và thiếu minh bạch quy trình.

### Gap 2: Thiếu Kiểm Chứng Ngoài (External Validation) & Nguy Cơ Rò Rỉ Dữ Liệu (Data Leakage)
* *Thực trạng*: Các công trình như Chicco & Jurman (2020) hay Ishaq et al. (2021) chỉ thực hiện trên tập dữ liệu UCI nhỏ (299 bệnh nhân) tại một trung tâm duy nhất ở Pakistan.
* *Nguy cơ Data Leakage*: Việc áp dụng kỹ thuật cân bằng dữ liệu SMOTE hoặc Imputation nếu thực hiện **trước khi chia tập Train/Test** sẽ làm rò rỉ thông tin từ tập Test vào tập Train, dẫn đến độ chính xác bị thổi phồng ảo ($92.62\%$). Cần kiểm chứng ngoài trên bộ dữ liệu lớn hơn như PhysioNet Zigong (2,008 bệnh nhân).

### Gap 3: Mâu Thuẫn Giữa Khả Năng Giải Thích (Interpretability) & Tính Khả Thi Lâm Sàng (Clinical Feasibility)
* *Thực trạng*: Các mô hình ML phức tạp (Deep Learning, XGBoost) bị xem là "hộp đen". Mặt khác, mô hình sử dụng hàng chục biến số không thực tế trong phòng cấp cứu.
* *Khoảng trống*: Mặc dù Chicco & Jurman (2020) chứng minh chỉ cần 2 biến (*Serum Creatinine* và *Ejection Fraction*) trên 299 bệnh nhân UCI, tính tổng quát của phát hiện này chưa từng được kiểm chứng trên tập dữ liệu đa trung tâm quy mô lớn hơn (>2,000 bệnh nhân).

### Gap 4: Bỏ Qua Tính Dị Biệt Phân Nhóm Bệnh Nhân (Subgroup Heterogeneity)
* *Thực trạng*: Hầu hết các nghiên cứu ML gộp chung tất cả bệnh nhân suy tim thành một tập "General HF" mà không đánh giá riêng trên từng phân nhóm LVEF (HFrEF, HFmrEF, HFpEF) hay theo tình trạng cấp tính/mãn tính, gây suy giảm độ tin cậy khi áp dụng lâm sàng.

### Gap 5: Bỏ Quên Yếu Tố Chuỗi Thời Gian Dọc (Longitudinal Sequential EHR Data)
* *Thực trạng*: Các mô hình hiện tại chủ yếu khai thác dữ liệu cắt ngang (cross-sectional) tại thời điểm nhập viện mà chưa tận dụng được lịch sử biến thiên theo thời gian của các chỉ số sinh hiệu trong các đợt tái khám.

---

## 3. Đóng Góp Dự Kiến Của Đề Tài (Proposed Contributions)

1. **Xây dựng Pipeline Tiền Xử Lý Chuẩn Tránh Data Leakage**: Tách biệt nghiêm ngặt Train/Test trước khi thực hiện Imputation, Rescaling và SMOTE.
2. **Thực hiện External Validation Đa Trung Tâm**: Kiểm chứng các mô hình huấn luyện trên UCI (299 ca) sang PhysioNet Zigong (2,008 ca) và ngược lại.
3. **Xác Minh Bộ Đặc Trưng Tối Giản (Minimal Clinical Feature Set)**: Kiểm chứng giả thuyết 2 biến (*Serum Creatinine* + *LVEF*) của Chicco & Jurman trên quy mô dữ liệu PhysioNet Zigong.
4. **Phân Tích Khả Năng Giải Thích Mô Hình (Explainable AI - XAI)**: Sử dụng **SHAP (SHapley Additive exPlanations)** để minh bạch hóa đóng góp của từng chỉ số sinh học đối với nguy cơ tử vong.
5. **Đánh Giá Toàn Diện Bằng Các Chỉ Số Chuẩn Y Khoa**: Ưu tiên PR-AUC, MCC, Brier Score (Calibration) bên cạnh ROC-AUC và Accuracy.
