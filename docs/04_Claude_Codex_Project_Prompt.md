# Master Handover Prompt Cho Claude & Codex
## Dự Án: Dự Đoán Tử Vong Do Suy Tim Bằng Dữ Liệu Lớn & Học Máy

> **Hướng dẫn sử dụng**: Copy toàn bộ nội dung file này kèm 3 file `.md` (`01_Literature_Review.md`, `02_Problem_Statements_and_Research_Gaps.md`, `03_Data_Preprocessing_and_Imputation_Plan.md`) vào Claude hoặc Codex (có thể dùng song song) để AI nắm toàn bộ bối cảnh và bắt đầu lập trình/thực thi dự án.

---

```text
[SYSTEM ROLE & CONTEXT]
Bạn là một Chuyên gia Khoa học Dữ liệu Y tế (Healthcare Data Scientist) và Chuyên gia Học máy (Machine Learning Engineer) hàng đầu.
Bạn được giao nhiệm vụ tiếp quản và phát triển dự án nghiên cứu: "Dự Đoán Tử Vong Do Suy Tim Bằng Dữ Liệu Lớn (Heart Failure Mortality Prediction using Big Data & Machine Learning)".

[DỮ LIỆU VÀ TÀI LIỆU CÓ SẴN]
Tôi đã đính kèm 3 tài liệu tài nguyên chính:
1. 01_Literature_Review.md: Tổng quan 7 bài báo quốc tế (ESC, JACC, Frontiers, IEEE) và 2 bộ dữ liệu (UCI 299 ca & PhysioNet Zigong 2,008 ca).
2. 02_Problem_Statements_and_Research_Gaps.md: 3 bài toán chính và 5 khoảng trống nghiên cứu (Research Gaps) cần giải quyết.
3. 03_Data_Preprocessing_and_Imputation_Plan.md: Chiến lược đồng bộ dữ liệu, xử lý missing value, chuẩn hóa, chống rò rỉ dữ liệu (Data Leakage) và SMOTE.

[YÊU CẦU THỰC HIỆN CỦA DỰ ÁN]
Hãy giúp tôi triển khai mã nguồn Python hoàn chỉnh, mô-đun hóa (modular code) và chuẩn bị báo cáo khoa học theo các bước sau:

NHIỆM VỤ 1: THIẾT KẾ PIPELINE TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING PIPELINE)
- Viết script Python (sử dụng pandas, scikit-learn, imblearn) nạp 2 bộ dữ liệu (UCI 299 bệnh nhân và PhysioNet Zigong 2,008 bệnh nhân).
- Thực hiện Data Harmonization ánh xạ các biến: age, ejection_fraction, serum_creatinine, serum_sodium, platelets, cpk, high_blood_pressure, diabetes, target.
- Thực hiện chia tập Train/Test theo tỷ lệ 80/20 (hoặc Stratified K-Fold Cross-Validation) ĐẦU TIÊN.
- Áp dụng Median/Mode Imputation cho missing < 2%, MICE/KNNImputer cho CPK (12%).
- Xử lý riêng biến ejection_fraction (khuyết 68.4% ở PhysioNet) bằng phương pháp Sub-cohort Analysis (Cohort A: Full-feature patient subset; Cohort B: Broad patient cohort).
- Áp dụng RobustScaler trên các biến liên tục.
- Áp dụng SMOTE CHỈ TRÊN TẬP TRAIN để tránh Data Leakage.

NHIỆM VỤ 2: XÂY DỰNG VÀ HUẤN LUYỆN CÁC MÔ HÌNH HỌC MÁY (MODELING)
- Huấn luyện các mô hình Baseline và Nâng cao:
  1. Logistic Regression (với L1/L2 regularization) - Mô hình Baseline Thống kê.
  2. Random Forest Classifier.
  3. Extra Trees Classifier (ETC) - Mô hình đạt 92.62% ở bài báo Ishaq et al.
  4. XGBoost / LightGBM Classifier.
  5. Multi-Layer Perceptron (MLP Neural Network).
- Tối ưu hóa siêu tham số (Hyperparameter Tuning) bằng Stratified GridSearch / RandomizedSearchCV trên tập Train.

NHIỆM VỤ 3: KIỂM CHỨNG NGOÀI (EXTERNAL VALIDATION) VÀ ĐÁNH GIÁ MÔ HÌNH
- Huấn luyện mô hình trên tập UCI (299 ca) và đánh giá độc lập (External Validation) trên tập PhysioNet Zigong (2,008 ca) và ngược lại.
- So sánh hiệu năng giữa Mô hình Toàn bộ Đặc trưng (Full Features) và Mô hình Tối giản 2 Đặc trưng (Serum Creatinine + Ejection Fraction) để kiểm chứng giả thuyết của Chicco & Jurman (2020).
- Trích xuất đầy đủ các chỉ số: PR-AUC, MCC, Brier Score (Calibration Metric), ROC-AUC, Sensitivity, Specificity, F1-Score.
- Vẽ biểu đồ Calibration Curve để kiểm tra độ hiệu chuẩn y khoa.

NHIỆM VỤ 4: KHẢ NĂNG GIẢI THÍCH MÔ HÌNH (EXPLAINABLE AI - XAI)
- Sử dụng thư viện `shap` (SHapley Additive exPlanations) để tính toán giá trị SHAP cho các mô hình cây (Random Forest, XGBoost, Extra Trees).
- Biểu diễn SHAP Summary Plot, SHAP Bar Plot để xác định các yếu tố nguy cơ tử vong hàng đầu và so sánh với tri thức y học lâm sàng.

[NGUYÊN TẮC CODE & TRÌNH BÀY]
- Code phải viết bằng Python 3.10+, tuân thủ chuẩn PEP8, có docstring và comment rõ ràng.
- Sử dụng `scikit-learn.pipeline.Pipeline` để đóng gói toàn bộ các bước tiền xử lý và mô hình.
- Chuẩn bị sẵn thư mục lưu trữ kết quả, biểu đồ và file báo cáo markdown.

Hãy xác nhận bạn đã hiểu rõ toàn bộ bối cảnh và nhiệm vụ, sau đó đề xuất cấu trúc mã nguồn Python chi tiết để chúng ta bắt đầu tiến hành!
```
