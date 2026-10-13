# Tổng Quan Tài Liệu Nghiên Cứu (Literature Review)
## Dự Đoán Tử Vong Do Suy Tim Bằng Dữ Liệu Lớn & Học Máy

---

## 1. Giới Thiệu Chung
Suy tim (Heart Failure - HF) là một hội chứng lâm sàng phức tạp mang tính toàn cầu, ảnh hưởng đến hơn 64 triệu người trên thế giới. Việc tiên lượng chính xác nguy cơ tử vong và tái nhập viện của bệnh nhân suy tim có ý nghĩa quan trọng trong việc phân tầng nguy cơ, hỗ trợ bác sĩ đưa ra phác đồ điều trị cá thể hóa. 

Báo cáo này tổng hợp và đánh giá hệ thống 7 công trình nghiên cứu xuất bản trên các tạp chí y khoa/khoa học máy tính hàng đầu (ESC Heart Failure, JACC, American Heart Journal, Frontiers in Cardiovascular Medicine, IEEE Access, BMC Medical Informatics and Decision Making) cùng 2 bộ dữ liệu lâm sàng chính (**UCI/Kaggle** và **PhysioNet Zigong**).

---

## 2. Phân Nhóm và Phân Tích Chi Tiết Các Tài Liệu Nghiên Cứu

### Nhóm 1: Dịch Tễ Học & Gánh Nặng Y Tế Toàn Cầu
1. **Shahim et al. (2023)** - *Global Public Health Burden of Heart Failure: An Updated Review* (ESC Heart Failure, Q1)
   - **Tóm tắt**: Cung cấp bức tranh toàn cảnh về dịch tễ học suy tim toàn cầu (~64 triệu người mắc, chi phí y tế dự kiến tăng 127% từ 2012 đến 2030 tại Mỹ). Cập nhật định nghĩa chuẩn hóa năm 2021 (ESC/AHA) phân loại suy tim theo phân suất tống máu LVEF:
     - **HFrEF** ($LVEF \le 40\%$): Suy tim giảm phân suất tống máu.
     - **HFmrEF** ($LVEF: 41 - 49\%$): Suy tim giảm nhẹ phân suất tống máu.
     - **HFpEF** ($LVEF \ge 50\%$): Suy tim bảo tồn phân suất tống máu.
     - **HFimpEF**: Suy tim cải thiện phân suất tống máu ($\ge 10\%$ tăng từ $LVEF \le 40\%$).
   - **Đóng góp**: Thiết lập cơ sở lý luận về gánh nặng y tế và sự dịch chuyển dịch tễ học từ HFrEF sang HFpEF.

2. **Kapelios et al. (2023)** - *Epidemiology, Clinical Characteristics and Cause-specific Outcomes in Heart Failure with Preserved Ejection Fraction* (Cardiac Failure Review / JACC, Q1)
   - **Tóm tắt**: Tập trung phân tích chuyên sâu về HFpEF. Bệnh nhân HFpEF thường là người cao tuổi, nữ giới, có tỷ lệ cao mắc các bệnh đồng mắc phức tạp (tăng huyết áp, rung nhĩ AF, bệnh thận mãn tính CKD, béo phì, tiểu đường). Tỷ lệ tử vong phi tim mạch (non-cardiovascular death) ở HFpEF cao hơn đáng kể so với HFrEF.
   - **Đóng góp**: Làm rõ sự dị biệt kiểu hình (phenotypic heterogeneity) giữa các phân nhóm suy tim.

---

### Nhóm 2: Đánh Giá Tổng Quan & Phân Tích Gộp (Meta-Analysis) về Học Máy Trong Suy Tim
3. **Olsen et al. (2020)** - *Clinical applications of machine learning in the diagnosis, classification, and prediction of heart failure* (American Heart Journal, Q1)
   - **Tóm tắt**: Tổng quan các ứng dụng của AI/ML trong tim mạch:
     - *Chẩn đoán*: Phân tích điện tâm đồ (ECG), siêu âm tim (TTE), dữ liệu hồ sơ sức khỏe điện tử (EHR).
     - *Phân loại kiểu hình (Phenomapping)*: Sử dụng thuật toán gom cụm không giám sát (Clustering: Hierarchical, K-Means) để tìm các phân nhóm bệnh nhân ngầm ẩn.
     - *Dự đoán tiên lượng*: Dự đoán tử vong và tái nhập viện 30 ngày.
   - **Cảnh báo kỹ thuật**: Nhấn mạnh hiện tượng "Lời nguyền chiều dữ liệu" (Curse of Dimensionality), nguy cơ Overfitting, mất cân bằng lớp (Class Imbalance) và khuyến nghị ưu tiên chỉ số **PR-AUC (Precision-Recall AUC)** và **MCC (Matthews Correlation Coefficient)** thay vì ROC-AUC khi số ca tử vong chiếm tỷ lệ thấp.

4. **Sun et al. (2022)** - *Comparing Machine Learning Models and Statistical Models for Predicting Heart Failure Events: A Systematic Review and Meta-Analysis* (Frontiers in Cardiovascular Medicine, Q2)
   - **Tóm tắt**: Tổng quan hệ thống PRISMA/CHARMS và phân tích gộp PROBAST trên **202 nghiên cứu mô hình thống kê** và **78 nghiên cứu mô hình ML**.
   - **Kết quả cốt lõi**:
     - C-index tổng hợp dự đoán tử vong: ML đạt **0.777** (95% CI: 0.752–0.803) vs Thống kê đạt **0.733** (95% CI: 0.724–0.742).
     - C-index tổng hợp dự đoán tái nhập viện: ML đạt **0.660** vs Thống kê đạt **0.678** (ML không vượt trội).
   - **Phát hiện quan trọng**:
     - Mô hình ML **không thể hiện sự vượt trội nhất quán** so with Logistic Regression/Cox Regression.
     - **Nguy cơ sai lệch rất cao (High Risk of Bias - PROBAST)**: $97\%$ mô hình ML vi phạm quy trình thống kê (kích thước mẫu nhỏ, thiếu bước hiệu chuẩn Calibration, thiếu kiểm chứng ngoài External Validation, và lạm dụng quá nhiều biến số gây khó khăn trên lâm sàng).

---

### Nhóm 3: Thực Nghiệm Dự Đoán Tiên Lượng Trên Dữ Liệu Bảng (Tabular Clinical Data)
5. **Chicco & Jurman (2020)** - *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone* (BMC Medical Informatics and Decision Making, Q2)
   - **Tóm tắt**: Phân tích tập dữ liệu gốc 299 bệnh nhân suy tim tại Pakistan (UCI/Kaggle). Thực hiện xếp hạng đặc trưng (Feature Ranking) bằng phương pháp biostatistics và Machine Learning (Random Forest).
   - **Kết quả cốt lõi**: Chứng minh rằng **chỉ với 2 chỉ số sinh hiệu chính là Serum Creatinine (Creatinine huyết thanh) và Ejection Fraction (Phân suất tống máu LVEF)**, mô hình ML (Random Forest, Gradient Boosting) đạt độ chính xác tiên lượng cao hơn (MCC > 0.41) so với việc sử dụng toàn bộ 12 thuộc tính lâm sàng.
   - **Đóng góp**: Đặt nền móng cho mô hình tiên lượng tối giản, khả thi trên lâm sàng cấp cứu.

6. **Ishaq et al. (2021)** - *Improving the Prediction of Heart Failure Patients' Survival Using SMOTE and Effective Data Mining Techniques* (IEEE Access, Q1)
   - **Tóm tắt**: Sử dụng kỹ thuật oversampling **SMOTE** để giải quyết mất cân bằng lớp trên bộ dữ liệu 299 bệnh nhân UCI, kết hợp thuật toán chọn lọc đặc trưng Random Forest (giữ lại 9 biến quan trọng).
   - **Kết quả cốt lõi**: Mô hình **Extra Trees Classifier (ETC)** đạt độ chính xác **92.62%** (Precision: 0.93, Recall: 0.93, F1-score: 0.93).
   - **Hạn chế**: Kích thước mẫu nhỏ (299 bệnh nhân), không có kiểm chứng ngoài. Cần kiểm tra kỹ cơ chế áp dụng SMOTE để tránh hiện tượng rò rỉ dữ liệu (Data Leakage).

7. **Zhang et al. (2021)** - *Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data (PhysioNet Zigong Dataset v1.3)* (Scientific Data, Q1)
   - **Tóm tắt**: Mô tả bộ dữ liệu lâm sàng hồi cứu gồm **2,008 bệnh nhân suy tim nhập viện** tại Bệnh viện Số 4 Tự Cống (Tứ Xuyên, Trung Quốc) từ 2016–2019 với **168 thuộc tính** (sinh hiệu, xét nghiệm máu, siêu âm tim, thuốc điều trị, biến tử vong/nhập viện).
   - **Đóng góp**: Cung cấp bộ dữ liệu bảng thực tế lớn, đa dạng để làm tập kiểm thử bên ngoài (External Validation) hoặc huấn luyện gộp.

8. **Slide Báo Cáo ĐH Bách Khoa Hà Nội (Group 43: Tchoua & Dadsi, Instructor: PhD. Hoàng Quang Huy)** - *Heart Failure Mortality Risk Stratification from Tabular Data*
   - **Tóm tắt**: Đề xuất pipeline kết hợp và đồng bộ hóa 2 bộ dữ liệu: Kaggle/UCI (299 bệnh nhân) và PhysioNet Zigong (2,008 bệnh nhân). Tiến hành phân tích EDA, xử lý dữ liệu khuyết (PhysioNet thiếu $68.4\%$ LVEF, $12\%$ CPK), chuẩn hóa đặc trưng và đánh giá mô hình Baseline (Logistic Regression, Random Forest) vs Improvement (SMOTE + Extra Trees).

---

## 3. Bảng Ma Trận Tổng Hợp Tài Liệu Nghiên Cứu (Literature Matrix)

| STT | Tên Bài Báo / Tài Liệu | Tác Giả & Năm | Tạp Chí / Nguồn | Loại Nghiên Cứu / Bộ Dữ Liệu | Đóng Góp Chính | Hạn Chế / Khoảng Trống | Citation Key |
|---|---|---|---|---|---|---|---|
| 1 | Global Public Health Burden of Heart Failure | Shahim et al. (2023) | ESC Heart Failure (Q1) | Tổng quan Dịch tễ học / Registries toàn cầu | Cập nhật định nghĩa ESC 2021, phân loại HFrEF, HFmrEF, HFpEF, HFimpEF. | Không đi sâu vào thuật toán ML thực nghiệm. | Shahim2023 |
| 2 | Epidemiology & Cause-specific Outcomes in HFpEF | Kapelios et al. (2023) | CFR / JACC (Q1) | Tổng quan Lâm sàng / Registries HFpEF | Phân tích sâu kiểu hình HFpEF, bệnh đồng mắc và tử vong phi tim mạch. | Thiếu các mô hình dự báo bằng thuật toán ML. | Kapelios2023 |
| 3 | Clinical applications of ML in diagnosis & prediction of HF | Olsen et al. (2020) | American Heart Journal (Q1) | Tổng quan ML Y khoa | Khung lý thuyết ML trong tim mạch; khuyến nghị dùng PR-AUC & MCC cho imbalanced data. | Bài tổng quan, không xây dựng pipeline cụ thể. | Olsen2020 |
| 4 | Comparing ML Models and Statistical Models for Predicting HF Events | Sun et al. (2022) | Front. Cardiovasc. Med. (Q2) | Meta-Analysis / 280 mô hình (202 Thống kê, 78 ML) | ML không vượt trội nhất quán so với Logistic Regression; 97% bài báo ML bị High Risk of Bias. | Thiếu phân tích sâu trên từng chuỗi dữ liệu thời gian. | Sun2022 |
| 5 | Improving Prediction of HF Survival Using SMOTE and Data Mining | Ishaq et al. (2021) | IEEE Access (Q1) | Thực nghiệm ML / UCI Dataset (299 bệnh nhân) | Kết hợp SMOTE + Extra Trees Classifier đạt Accuracy 92.62%. | Mẫu nhỏ (299 ca), nguy cơ rò rỉ dữ liệu khi làm SMOTE. | Ishaq2021 |
| 6 | Machine learning can predict survival from creatinine & ejection fraction alone | Chicco & Jurman (2020) | BMC Med. Inform. Decis. Mak. (Q2) | Thực nghiệm ML & Biostatistics / UCI Dataset (299 bệnh nhân) | Khám phá ra Serum Creatinine & Ejection Fraction đủ để dự đoán tử vong chính xác hơn 12 biến. | Mẫu nhỏ đơn trung tâm, thiếu kiểm chứng ngoài. | Chicco2020 |
| 7 | Electronic healthcare records for hospitalized HF patients (Zigong) | Zhang et al. (2021) | Scientific Data / PhysioNet (Q1) | Mô tả Dữ liệu / PhysioNet Zigong (2,008 bệnh nhân) | Bộ dữ liệu suy tim bảng lớn (2,008 ca, 168 biến số). | Dữ liệu khuyết nhiều ở một số biến siêu âm tim ($68.4\%$ LVEF). | Zhang2021 |
| 8 | Heart Failure Mortality Risk Stratification from Tabular Data | Group 43 - HUST (2026) | Slide Báo cáo ĐH Bách Khoa Hà Nội | Thực nghiệm ML / Ghép UCI (299) & PhysioNet (2,008) | Pipeline gộp 2 dataset, xử lý mất cân bằng và thử nghiệm Logistic Regression, RF, Extra Trees. | Đang trong giai đoạn hoàn thiện pipeline tiền xử lý. | HUST2026 |
