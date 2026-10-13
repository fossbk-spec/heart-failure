# BÁO CÁO GIẢI ĐỀ KIỂM TRA TỰ LUẬN & ĐỊNH HƯỚNG TRIỂN KHAI DỰ ÁN: HEART FAILURE SURVIVAL PREDICTION USING MACHINE LEARNING

## TỔNG QUAN DỰ ÁN
- **Tên dự án:** Heart Failure Survival Prediction using Machine Learning
- **Mục tiêu:** Xây dựng mô hình Học máy dự đoán nguy cơ tử vong của bệnh nhân suy tim từ dữ liệu lâm sàng tại thời điểm đánh giá ban đầu, đảm bảo độ tin cậy, không rò rỉ dữ liệu (Zero Data Leakage) và có tính giải thích lâm sàng cao.
- **Bộ dữ liệu chuẩn:** Heart Failure Clinical Records (UCI #519; Ahmad et al. 2017; Chicco & Jurman 2020) — 299 bệnh nhân, 13 thuộc tính.

---

## PHẦN I: PROBLEM STATEMENT AND RISK-COST ANALYSIS (2.0 điểm)

### Câu 1.1: Phát biểu bài toán lâm sàng & Không gian dữ liệu
1. **Tập bệnh nhân mục tiêu (Target Population):**
   - Bệnh nhân suy tim ≥ 40 tuổi có rối loạn chức năng tâm thu thất trái, NYHA III–IV, đang được theo dõi sau nhập viện — nhóm cần phân tầng nguy cơ để quyết định mức độ theo dõi, tối ưu điều trị nội khoa, cân nhắc thiết bị/ghép tim hoặc chăm sóc giảm nhẹ.
2. **Không gian đặc trưng đầu vào ($X$):**
   - $X \in \mathbb{R}^{11}$ gồm 11 thuộc tính lâm sàng có sẵn **tại thời điểm đánh giá ban đầu**:
     - Liên tục: `age` (tuổi), `creatinine_phosphokinase` (CPK, mcg/L), `ejection_fraction` (LVEF, %), `platelets` (kiloplatelets/mL), `serum_creatinine` (mg/dL), `serum_sodium` (mEq/L).
     - Nhị phân (0/1): `anaemia`, `diabetes`, `high_blood_pressure`, `sex` (1 = nam), `smoking`.
   - **Không thuộc $X$:** `patient_id` (định danh, do `data_loader` thêm vào) và `time` (số ngày theo dõi — chỉ biết SAU khi có kết cục, xem Câu 4.2).
3. **Nhãn đầu ra ($Y$):**
   - $Y = $ `DEATH_EVENT` $\in \{0, 1\}$ (0: còn sống đến hết theo dõi, 1: tử vong trong thời gian theo dõi).
4. **Phân loại bài toán Học máy:**
   - **Supervised Learning** — **Binary Classification**. (Cách tiếp cận thay thế: phân tích sống còn với `time` là biến kết cục — Cox PH, như Ahmad et al. 2017.)

### Câu 1.2: Phân tích rủi ro bất đối xứng & Lựa chọn độ đo
1. **Rủi ro bất đối xứng ($Cost(FN) \gg Cost(FP)$):**
   - **False Negative (FN):** dự đoán "sống" nhưng bệnh nhân tử vong → bỏ lỡ cơ hội tăng cường theo dõi, chỉnh liều thuốc, can thiệp thiết bị hoặc thảo luận chăm sóc giảm nhẹ kịp thời. Hậu quả có thể là tử vong có thể phòng tránh.
   - **False Positive (FP):** dự đoán "tử vong" nhưng bệnh nhân sống → tăng tần suất tái khám, xét nghiệm, chi phí và lo âu.
   - **Kết luận:** ưu tiên hàng đầu là giảm FN, nhưng vẫn phải giữ Specificity ở mức chấp nhận được — nếu không, mô hình chỉ là "gắn cờ tất cả".
2. **Lựa chọn độ đo:**
   - **Không dùng Accuracy đơn lẻ:** mô hình luôn dự đoán "sống" đạt Null Accuracy **67.9%** với Recall = 0.
   - **Bộ độ đo bắt buộc** (thứ tự ưu tiên): **Recall > PR-AUC > F2 > ROC-AUC > Specificity**, kèm Brier score (hiệu chỉnh xác suất).
   - **Lưu ý riêng cho bộ dữ liệu này:** tỷ lệ dương 32.1% nên quy tắc tầm thường "gắn cờ mọi bệnh nhân" đã đạt **F2 ≈ 0.70** (`f2_flag_all`). Vì vậy F2 không đủ để chọn ngưỡng; dự án dùng **ràng buộc Recall ≥ 0.80** trên tập validation (`--min-recall 0.8`), rồi đọc Recall cùng Specificity và PR-AUC (baseline PR-AUC ngẫu nhiên = 0.32).

---

## PHẦN II: DEEP LITERATURE REVIEW — CORE FOCUS (5.0 điểm)

### Câu 2.1: Ma trận y văn & Tiêu chuẩn thẩm định
1. **Tiêu chuẩn lựa chọn bài báo:**
   - SJR Q1/Q2 (Scopus / Web of Science); xuất bản 2020–2026.
   - Hội nghị hàng đầu (MICCAI, MLHC, NeurIPS, IEEE EMBC) phải qua phản biện kín; preprint (arXiv, medRxiv) chỉ tham khảo bổ trợ.
   - **Ngoại lệ có chủ đích:** Ahmad et al. (2017) nằm ngoài khung thời gian nhưng là **bài báo gốc của bộ dữ liệu** → bắt buộc trích dẫn.
2. **8 trường thông tin bắt buộc:** (1) Tác giả & Năm; (2) Tập bệnh nhân; (3) Nguồn dữ liệu & Cỡ mẫu; (4) Đặc trưng chính; (5) Mô hình ML/DL; (6) Giao thức chia dữ liệu; (7) Độ đo & Kết quả chính; (8) Hạn chế & Khoảng trống.
3. **Nguyên tắc truy vết (Traceability):** mọi số liệu phải ghi vị trí trong văn bản gốc (trang/bảng). Các số liệu dưới đây hiện được trích từ **abstract PubMed** (đã ghi PMID/DOI trong `docs/literature/`); nhóm phải bổ sung số trang/bảng sau khi đặt PDF vào thư mục SSoT.

#### Bảng tổng quan Ma trận Y văn (Literature Matrix):
| Tác giả & Năm | Tập bệnh nhân | Nguồn dữ liệu & Cỡ mẫu | Đặc trưng chính | Mô hình | Giao thức chia dữ liệu | Độ đo & Kết quả (abstract) | Hạn chế & Khoảng trống |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ahmad et al. (2017)**, PLoS ONE | Suy tim ≥40 tuổi, NYHA III–IV | Faisalabad, Pakistan, 04–12/2015 (299 BN) | Tuổi, EF, creatinine, natri, thiếu máu, huyết áp... | Cox PH, Kaplan–Meier, nomogram | Bootstrapping nội bộ | Yếu tố có ý nghĩa: tuổi, chức năng thận, huyết áp, EF, thiếu máu | Một trung tâm; không kiểm chứng ngoài; mô hình thống kê cổ điển |
| **Chicco & Jurman (2020)**, BMC MIDM | Như trên | Cùng bộ 299 BN | Toàn bộ 12 đặc trưng; mô hình 2 đặc trưng (creatinine + EF) | Nhiều classifier ML + xếp hạng đặc trưng | *Cần đối chiếu PDF* | Creatinine & EF là 2 đặc trưng quan trọng nhất; mô hình 2 đặc trưng tốt hơn toàn bộ | Phân tích có đưa thời gian theo dõi vào → nguy cơ rò rỉ thời gian; cỡ mẫu nhỏ |
| **Wilstrup & Cave (2022)**, BMC MIDM | Như trên | Cùng bộ 299 BN | exp(tuổi), 1/EF, 1/creatinine | QLattice (symbolic regression) + Cox PH | *Cần đối chiếu PDF* | Cox trên biến đã biến đổi tốt hơn Cox trên biến gốc | Cùng hạn chế về dữ liệu; chỉ so sánh với Cox |
| **Li et al. (2022)**, JMIR | Suy tim tại ICU | eICU-CRD (2,798 BN; tử vong 9.97%) | Dữ liệu 24h đầu ICU (BUN, ...) | XGBoost vs 3 mô hình ML; SHAP | Chia ngẫu nhiên 70/30 | XGBoost AUC 0.824 (95% CI 0.777–0.871) | Không kiểm chứng ngoài; chưa rõ gom nhóm theo bệnh nhân |
| **Chen et al. (2023)**, Front Cardiovasc Med | Suy tim sung huyết tại ICU | MIMIC-IV (11,156) → eICU-CRD (9,837) | 17 đặc trưng (LASSO): APS III, tuổi, SOFA... | XGBoost vs LR vs GWTG-HF; SHAP | **Kiểm chứng ngoài** giữa 2 CSDL | Ngoài: AUC 0.771 (0.757–0.784), Brier 0.100 | Dữ liệu ICU, khó áp dụng ngoại trú |
| **Tanaka et al. (2024)**, ESC Heart Fail | Suy tim cấp | KCHF registry, Nhật (4,040 BN; tử vong 6.3%) | Thông tin ngày nhập viện đầu | CART, RF, XGBoost; SHAP + phân cụm | Chia 4:1 (3,232/808) | Tử vong: XGBoost AUC 0.816 vs RF 0.755 vs CART 0.683 | Chỉ kết cục nội viện; một quốc gia |
| **Petmezas et al. (2025)**, J Biomed Inform | Suy tim | Siêu âm tim + nhân khẩu học (*cỡ mẫu: đối chiếu PDF*) | 7 đặc trưng chọn lọc | 7 mô hình ML; Extra-Trees + tương quan phi tuyến; SHAP | *Cần đối chiếu PDF* | Giữ/cải thiện hiệu năng khi giảm 80% đặc trưng | Chọn đặc trưng phải nằm trong CV để tránh rò rỉ |

### Câu 2.2: Phân tích khoảng trống nghiên cứu & Mô hình Baseline
1. **Khoảng trống nghiên cứu chính:**
   - **Rò rỉ thời gian (Temporal leakage):** biến `time` của bộ 299 BN là thời gian theo dõi, bị cắt ngắn bởi chính biến cố tử vong. Đưa `time` vào X làm ROC-AUC (grouped 5-fold CV, Random Forest) tăng giả tạo từ **0.79 lên 0.92** (notebook 02).
   - **Rò rỉ do tiền xử lý/oversampling trước khi chia dữ liệu** và chia ngẫu nhiên không gom nhóm theo bệnh nhân.
   - **Cỡ mẫu nhỏ, một trung tâm, không kiểm chứng ngoài** đối với bộ 299 BN — trái ngược với Chen et al. (2023).
   - **Thiếu tính giải thích lâm sàng (XAI)** ở các mô hình ensemble.
   - **Báo cáo độ đo không phù hợp:** Accuracy/F1 tại ngưỡng 0.5, bỏ qua Specificity và calibration.
2. **Mô hình cơ sở (Baseline Models):**
   - **Logistic Regression** (tuyến tính, đọc được odds ratio).
   - **Cox Proportional Hazards** (Ahmad et al. 2017) — mô hình thống kê kinh điển cho dữ liệu sống còn.
   - **Mô hình 2 đặc trưng serum creatinine + ejection fraction** (Chicco & Jurman 2020).
   - Thang điểm lâm sàng tham chiếu cho suy tim: Seattle Heart Failure Model, MAGGIC (không áp dụng trực tiếp được vì bộ dữ liệu thiếu nhiều biến cần thiết như NYHA từng bệnh nhân, thuốc, BNP).

### Câu 2.3: Quản lý trích dẫn & Thư mục SSoT
1. **Thư mục Single Source of Truth (SSoT):** `data/SSoT_pdf_references/` (danh sách tệp cần tải và DOI ở README của thư mục).
2. **Quy ước đặt tên:** `[Năm]_[HọTácGiả]_[TênBàiBáo_ViếtTắt].pdf`, ví dụ `2020_Chicco_MLSurvivalHFCreatinineEF.pdf`.
3. **Chế tài:** trừ điểm nặng hoặc hủy kết quả nếu bịa đặt nguồn, trích dẫn không có PDF trong SSoT, hoặc số liệu sai lệch so với văn bản gốc.

---

## PHẦN III: EXPLORATORY DATA ANALYSIS - EDA (3.0 điểm)

*Mọi số liệu dưới đây được tái tạo bằng `make eda` (`scripts/run_eda.py`, `src/eda.py`).*

### Câu 3.1: Đặc tả thuộc tính dữ liệu & Thiên vị dịch tễ
1. **Đặc tả:** 299 bệnh nhân, 13 cột (11 đặc trưng, 1 thời gian theo dõi `time`, 1 nhãn `DEATH_EVENT`); không có cột định danh trong file gốc. Chi tiết: `docs/literature/03-uci-heart-failure-clinical-records-dataset.md`.
2. **Phân tầng nguy cơ & thiên vị:**
   - **Tuổi:** tỷ lệ tử vong tăng đơn điệu theo nhóm tuổi: 23.4% (40–49) → 24.4% (50–59) → 29.0% (60–69) → 38.5% (70–79) → **72.0% (≥ 80)**.
   - **Phân suất tống máu (phân loại ESC 2021):** HFrEF (≤ 40%, 219 BN) tử vong **35.2%**; HFmrEF (41–49%, 20 BN) 25.0%; HFpEF (≥ 50%, 60 BN) 23.3%.
   - **Creatinine huyết thanh:** ≤ 1.2 mg/dL: 21.2%; 1.5–2.0: **62.2%**; ≥ 2.0: **62.9%** (nguy cơ tương đối ≈ 2.9–3.0 lần).
   - **Natri huyết thanh:** hạ natri máu (< 135 mEq/L, 83 BN) tử vong **50.6%** so với 25.0% ở nhóm bình thường (≈ 2.0 lần).
   - **Giới tính:** nam 64.9% / nữ 35.1% — mất cân đối đại diện, dù tỷ lệ tử vong gần như bằng nhau (32.0% vs 32.4%). Mô hình có thể kém tin cậy hơn ở nhóm nữ.
   - **Tương quan point-biserial với nhãn:** serum_creatinine +0.29, ejection_fraction −0.27, age +0.25, serum_sodium −0.20; các biến nhị phân (diabetes, sex, smoking) gần 0.

### Câu 3.2: Thách thức dữ liệu y tế
1. **Mất cân bằng lớp:** 203 sống (67.9%) / 96 tử vong (32.1%) — mức vừa phải, nhưng tập test 20% chỉ còn 19 ca tử vong.
   - **Sample-level vs Patient-level:** bộ này có đúng 1 bản ghi/bệnh nhân nên hai mức trùng nhau; `class_balance_report` vẫn tính cả hai để phát hiện ngay khi áp dụng cho EHR có nhiều lần khám/bệnh nhân.
2. **Chất lượng dữ liệu:**
   - Không có NaN, nhưng **nội suy ẩn**: `platelets = 263358.03` (= trung bình cột) lặp 25 lần; `creatinine_phosphokinase = 582` lặp 47 lần.
   - Các xét nghiệm lệch phải mạnh (CPK tối đa 7,861; creatinine tối đa 9.4) → biến đổi `log1p`.
   - **Biến rò rỉ `time`:** trung vị 172 ngày (sống) vs 44.5 ngày (tử vong); AUC đơn biến 0.84 > mọi đặc trưng hợp lệ (cao nhất 0.73).
3. **Chế tài mã nguồn EDA:** các hàm EDA đóng gói, tái sử dụng, chạy end-to-end không lỗi — được kiểm thử trong `tests/test_eda.py`.

---

## PHẦN IV: ADVANCED PREPROCESSING & ZERO DATA LEAKAGE (+2.0 điểm)

### Câu 4.1: Quy trình tiền xử lý dữ liệu bảng/EHR
1. **Nội suy:** dữ liệu gốc không có NaN; vẫn đặt `SimpleImputer(median)` (mặc định) hoặc `KNNImputer` trong pipeline để chịu được bản ghi thiếu khi triển khai. Median là mặc định vì KNN chạy trước bước chuẩn hóa sẽ bị `platelets` (~10⁵) chi phối khoảng cách.
2. **Mã hóa:** mọi biến phân loại đều đã là 0/1 → giữ nguyên, không cần One-Hot.
3. **Biến đổi & chuẩn hóa:** `log1p` cho `creatinine_phosphokinase`, `serum_creatinine`, `platelets` (giảm độ lệch, phù hợp kết quả biến đổi phi tuyến của Wilstrup & Cave 2022); sau đó **StandardScaler** cho 6 biến liên tục (tắt với mô hình cây).
4. **Cân bằng lớp:** **SMOTE** đặt trong `imblearn.pipeline.Pipeline` (và trong từng base learner của Stacking); đối chứng bằng `class_weight="balanced"` (`--no-smote --class-weight balanced`).

### Câu 4.2: Kỷ luật chống rò rỉ dữ liệu tuyệt đối (Zero Data Leakage)
1. **Tác hại của chia ngẫu nhiên cấp bản ghi:** khi một bệnh nhân có nhiều bản ghi, mô hình "học thuộc" bệnh nhân đó → điểm test cao giả tạo. Bộ 299 BN có 1 bản ghi/BN, nhưng pipeline vẫn chia theo `patient_id` để dùng được cho dữ liệu EHR thật (được kiểm thử với dữ liệu tổng hợp nhiều bản ghi/BN).
2. **Chia theo bệnh nhân:** `patient_level_split` (train/val/test = 191/48/60 BN) phân tầng theo `DEATH_EVENT`, gom nhóm theo `patient_id`; CV dùng `StratifiedGroupKFold`.
3. **Fit-on-Train-Only:** imputer, `log1p`, scaler và SMOTE chỉ `fit` trên train; validation/test chỉ `.transform()`. Ngưỡng quyết định chọn trên validation, không bao giờ trên test; trong CV, ngưỡng chọn trên dự đoán out-of-fold của vòng CV trong.
4. **Loại biến hậu kết cục (`time`):** biến chỉ biết sau khi có kết cục không được là đầu vào của mô hình dự báo tại thời điểm ban đầu. `split_features_target` loại `time`; `tests/test_data_leakage.py` khẳng định `time` không có trong X và không được preprocessor tham chiếu.
5. **Mã kiểm chứng (rút gọn từ `src/preprocessing.py`):**

```python
from src import config
from src.data_loader import load_raw_data, split_features_target
from src.models import get_model
from src.preprocessing import (
    assert_no_fit_on_holdout, make_full_pipeline, patient_level_split, verify_zero_data_leakage,
)

df = load_raw_data()                       # thêm patient_id = 1..299
split = patient_level_split(df)            # phân tầng theo DEATH_EVENT, gom nhóm theo patient_id
verify_zero_data_leakage(split.train, split.val, split.test)   # raise DataLeakageError nếu trùng ID

X_train, y_train = split_features_target(split.train)          # loại patient_id và time
X_test, _ = split_features_target(split.test)
assert "time" not in X_train.columns

pipeline = make_full_pipeline(get_model("logistic_regression"))  # imputer -> log1p -> scaler -> SMOTE -> LR
pipeline.fit(X_train, y_train)                                    # mọi bước fit CHỈ trên train
assert_no_fit_on_holdout(pipeline, X_test)                        # trạng thái không đổi khi transform test
```

---

## KẾT QUẢ THỰC NGHIỆM (tái tạo bằng `make train` / `make train-recall80`)

Grouped 5-fold CV trên train+val, ngưỡng theo ràng buộc Recall ≥ 0.80:

| Mô hình | Recall | Specificity | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.86 ± 0.07 | 0.49 ± 0.15 | 0.63 ± 0.03 | 0.77 ± 0.04 |
| Random Forest | 0.79 ± 0.15 | 0.68 ± 0.14 | 0.67 ± 0.04 | 0.78 ± 0.05 |
| Gradient Boosting | 0.83 ± 0.13 | 0.46 ± 0.14 | 0.63 ± 0.11 | 0.74 ± 0.06 |
| Stacking Ensemble | 0.78 ± 0.10 | 0.62 ± 0.14 | 0.63 ± 0.11 | 0.76 ± 0.08 |

- Random Forest cho điểm cân bằng tốt nhất (PR-AUC 0.67, Specificity 0.68 ở Recall ≈ 0.8); Stacking không vượt trội rõ rệt trên bộ dữ liệu nhỏ này.
- Với chính sách F2 mặc định, Recall CV đạt 0.84–0.91 nhưng Specificity chỉ 0.27–0.45 — minh chứng cho cảnh báo ở Câu 1.2.
- **XAI:** SHAP (Random Forest) xếp `serum_creatinine` và `ejection_fraction` ở vị trí 1–2, tái lập kết luận của Chicco & Jurman (2020). Odds ratio (Logistic Regression, trên mỗi 1 SD sau log1p): serum creatinine **3.7** (95% CI 2.1–6.6), ejection fraction **0.43** (0.28–0.67) — CI chỉ mang tính tham khảo vì hệ số học trên dữ liệu đã SMOTE.

---

## KIẾN TRÚC PROJECT VÀ CẤU TRÚC REPOSITORY GITHUB

```
heart-failure/
│
├── .github/workflows/ci_cd_pipeline.yml   # Lint + test + Zero-Leakage audit gate
│
├── data/
│   ├── raw/                               # heart_failure_clinical_records_dataset.csv (bundled)
│   ├── processed/                         # Dữ liệu đã qua tiền xử lý (không commit)
│   └── SSoT_pdf_references/               # PDF các bài báo y văn (không commit)
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_imbalance.ipynb   # gồm thí nghiệm rò rỉ `time`
│   └── 03_model_training_and_evaluation.ipynb
│
├── src/
│   ├── config.py           # schema, LEAKAGE_COLUMNS, tham số thực nghiệm
│   ├── data_loader.py      # nạp dữ liệu, thêm patient_id, kiểm tra schema
│   ├── eda.py              # phân tầng LVEF / creatinine / natri, báo cáo rò rỉ
│   ├── preprocessing.py    # chia theo bệnh nhân, pipeline chống rò rỉ
│   ├── models.py           # LR, RF, GB, SVM, KNN, NB, XGBoost, Stacking
│   ├── evaluation.py       # Recall, PR-AUC, F2, f2_flag_all, chọn ngưỡng, grouped CV
│   ├── explainability.py   # SHAP, permutation importance, odds ratio
│   └── train.py            # CLI huấn luyện end-to-end
│
├── tests/                  # 99 unit test, gồm bộ kiểm thử chống rò rỉ
├── docs/
│   ├── DETAILED_EXAM_REPORT.md
│   └── literature/         # đặc tả bộ dữ liệu + 7 tóm tắt bài báo
├── requirements.txt
└── README.md
```
