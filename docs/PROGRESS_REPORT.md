# BÁO CÁO TỔNG HỢP CÔNG VIỆC — DỰ ÁN HEART FAILURE

**Dự án:** Dự đoán tử vong ở bệnh nhân suy tim bằng Machine Learning
**Repository:** https://github.com/fossbk-spec/heart-failure
**Ngày lập báo cáo:** 2026-10-13
**Công cụ hỗ trợ:** Claude Code và Codex (chỉ dùng trong quá trình phát triển; không xuất hiện với tư cách contributor trên GitHub)
**Kế hoạch tham chiếu:** [`docs/ACTION_PLAN_FOR_CLAUDE.md`](ACTION_PLAN_FOR_CLAUDE.md) v1.1

---

## 1. Tóm tắt

| Hạng mục | Trạng thái |
|---|---|
| Khởi tạo dự án theo template `stroke-prediction` | Hoàn thành |
| Gỡ ghi nhận Claude khỏi lịch sử Git (cả `stroke-prediction` và `heart-failure`) | Hoàn thành |
| Tài liệu nghiên cứu (tổng quan tài liệu, bài toán, tiền xử lý, 8 tóm tắt bài báo) | Hoàn thành |
| Kế hoạch hành động v1.1 | Hoàn thành |
| Giai đoạn 0 — Sửa nền (MCC, kiểm toán scaler) | Hoàn thành |
| Giai đoạn 1 — Mở rộng mô hình (Extra Trees, MLP) | Hoàn thành |
| Giai đoạn 2 — Mô hình tối giản 2 biến (Chicco & Jurman 2020) | Hoàn thành |
| Giai đoạn 3 — Tiếp nhận và đồng bộ dữ liệu Zigong | Hoàn thành |
| Giai đoạn 4 — Kiểm chứng ngoài UCI → Zigong (phương án A) | Hoàn thành |
| Giai đoạn 5 — Biểu đồ 300 DPI và `references.bib` | Chưa làm |
| Giai đoạn 6 — Tuỳ chọn RobustScaler | Chưa làm |
| Giai đoạn 7 — Phân tích sống còn (Cox, tuỳ chọn) | Chưa làm |

**Trạng thái kỹ thuật hiện tại:**
- 172 unit test đều qua (~25 giây), ruff không báo lỗi.
- CI chạy toàn bộ pipeline trên dữ liệu giả lập.
- Repo không chứa file dữ liệu Zigong nào.

**Kết luận khoa học chính:**
1. Trên UCI (299 bệnh nhân), mô hình tốt nhất đạt ROC-AUC CV khoảng 0.78–0.83. Hai biến creatinine huyết thanh và phân suất tống máu (EF) cho kết quả không kém 11 biến.
2. Biến `time` (thời gian theo dõi) gây rò rỉ dữ liệu: nếu đưa vào, ROC-AUC tăng giả tạo từ 0.79 lên 0.92. Biến này đã bị loại khỏi tập đặc trưng.
3. Khi đem mô hình UCI sang Zigong (Trung Quốc, 2.008 lượt nhập viện):
   - khả năng phân biệt chỉ ở mức khiêm tốn (ROC-AUC 0.59–0.79);
   - hiệu chuẩn thất bại hoàn toàn: nguy cơ dự đoán cao gấp khoảng 16 lần thực tế;
   - vì vậy không được dùng mô hình trên quần thể mới nếu chưa hiệu chỉnh lại.

---

## 2. Dữ liệu

### 2.1. UCI Heart Failure Clinical Records (dữ liệu phát triển)
- **Nguồn:** UCI ML Repository #519 (Ahmad et al. 2017; Chicco & Jurman 2020), giấy phép CC BY 4.0. File được đóng gói sẵn trong `data/raw/`.
- **Quy mô:** 299 bệnh nhân, 96 ca tử vong (32,1%), không có giá trị thiếu, kiểm tra toàn vẹn bằng sha256.
- **Đặc trưng:** 11 biến:
  - 6 biến số: tuổi, CPK, EF, tiểu cầu, creatinine, natri;
  - 5 biến nhị phân: thiếu máu, tiểu đường, tăng huyết áp, giới tính, hút thuốc.
- **Nhãn:** `DEATH_EVENT`.
- **ID bệnh nhân:** dữ liệu gốc không có ID, nên `data_loader` tự gán `patient_id` từ 1 đến 299 để chia tập theo bệnh nhân.

### 2.2. PhysioNet Heart Failure Zigong v1.3 (dữ liệu kiểm chứng ngoài)
- **Nguồn:** Zhang et al. 2021, bệnh viện Zigong (Trung Quốc), giai đoạn 2016–2019.
- **Quyền truy cập:** hạn chế (PhysioNet Restricted Health Data License 1.5.0). Nhóm đã có tài khoản PhysioNet được xác thực và đã ký DUA.
- **Quy mô:** 2.008 dòng × 167 cột. Mỗi dòng là một lượt nhập viện và không có ID bệnh nhân; khoảng 7% là lượt nhập viện lặp lại.
- **Tử vong:** 37 ca trong 28 ngày, 42 ca trong 90 ngày (2,1%), 57 ca trong 180 ngày.
- **Hạn chế:**
  - không có biến tăng huyết áp và hút thuốc;
  - LVEF thiếu ở 68,4% lượt nhập viện, và việc thiếu không ngẫu nhiên;
  - đơn vị CK chưa chắc chắn (trung vị 91, so với 259 ở UCI), vì file từ điển biến tải về bị lỗi (0 byte);
  - 5 dòng có nơi chuyển đến khi ra viện là "Died" nhưng nhãn tử vong 3 tháng lại bằng 0.
- **Quy định bảo mật:**
  - Dữ liệu chỉ nằm trong `data/raw/zigong/`. Thư mục này được git bỏ qua và không bao giờ được commit.
  - Không đưa dữ liệu lên CI, Colab hay Kaggle công khai.
  - Mọi kết quả công bố chỉ ở dạng tổng hợp.

---

## 3. Nguyên tắc Zero Data Leakage (áp dụng xuyên suốt)

1. **Chia tập theo bệnh nhân:** train/val/test gồm 191/48/60 bệnh nhân, phân tầng theo nhãn. Cross-validation dùng `StratifiedGroupKFold` 5 fold.
2. **Loại biến hậu kết cục:** `time` nằm trong `LEAKAGE_COLUMNS`. Có test kiểm tra rằng biến này không bao giờ lọt vào X.
3. **Chỉ fit trên tập train:** imputer, `log1p`, scaler và SMOTE chỉ học trên train. Tập validation và test chỉ gọi `.transform()`.
4. **SMOTE nằm trong Pipeline:** dùng `imblearn.Pipeline`, kể cả trong từng mô hình cơ sở của stacking.
5. **Kiểm toán tự động:**
   - `assert_no_fit_on_holdout()` và `verify_zero_data_leakage()` kiểm tra trạng thái đã fit của các bước (mean_, scale_, center_, lambdas_, statistics_ …);
   - Giai đoạn 0 mở rộng kiểm toán này sang RobustScaler và PowerTransformer.
6. **Kiểm chứng ngoài:** có test spy xác nhận mọi lần `fit` chỉ nhận dòng của UCI, không dòng Zigong nào.

---

## 4. Các công việc đã thực hiện

### 4.1. Khởi tạo dự án
- Dựng cấu trúc theo template `stroke-prediction`, gồm:
  - mã nguồn: `src/`, `scripts/`, `tests/`, `notebooks/` (01 EDA, 02 tiền xử lý và mất cân bằng, 03 huấn luyện và đánh giá);
  - công cụ: `Makefile`, CI GitHub Actions;
  - tài liệu: README có nút mở notebook trên Colab và Kaggle.
- Các module chính:
  - `config.py`: schema và các cột rò rỉ;
  - `data_loader.py`: tải dữ liệu, kiểm tra schema, sinh dữ liệu giả lập;
  - `preprocessing.py`: pipeline chống rò rỉ, AdaptiveSMOTE;
  - `models.py`, `evaluation.py`: Recall, PR-AUC, F2, MCC, chọn ngưỡng, grouped CV;
  - `explainability.py`: SHAP, permutation importance, odds ratio;
  - `eda.py`, `train.py`.
- Báo cáo chi tiết bài thi: [`DETAILED_EXAM_REPORT.md`](DETAILED_EXAM_REPORT.md).
- Thêm 8 tóm tắt tài liệu trong `docs/literature/` (03–10).
- Đã chạy sẵn 3 notebook bằng kernel của venv.

### 4.2. Quản trị repository
- Viết lại lịch sử Git của `stroke-prediction`, xoá các dòng `Co-Authored-By: Claude`, rồi force-push.
- Tạo repo công khai `fossbk-spec/heart-failure`. Mọi commit đều không có ghi nhận AI.
- Commit các tài liệu nhóm:
  - `01_Literature_Review.md`;
  - `02_Problem_Statements_and_Research_Gaps.md`;
  - `03_Data_Preprocessing_and_Imputation_Plan.md`;
  - `04_Claude_Codex_Project_Prompt.md`;
  - `preprocessing_pipeline.py`.

### 4.3. Kế hoạch hành động v1.1
Đã rà soát `ACTION_PLAN_FOR_CLAUDE.md` v1.0 và nâng lên v1.1 với các điểm chính:
- Chọn kết cục **tử vong 90 ngày** để đồng bộ UCI với Zigong. Quy tắc gán nhãn cho UCI:
  - nhãn 1 nếu tử vong trong vòng 90 ngày;
  - nhãn 0 nếu còn sống ở ngày 90 hoặc tử vong sau ngày 90;
  - loại bỏ bệnh nhân bị kiểm duyệt (mất theo dõi) khi còn sống trước ngày 90.
- Không impute LVEF cho Zigong.
- Chia lộ trình thành 8 giai đoạn (0–7) với checklist nghiệm thu.

### 4.4. Giai đoạn 0 — Sửa nền
- Thêm chỉ số **MCC** và `f2_flag_all`. Chỉ số sau là điểm F2 của quy tắc tầm thường "gắn cờ tất cả", khoảng 0.70 với tỉ lệ tử vong 32%.
- Thêm chính sách ngưỡng `--min-recall 0.8`. Lý do: tối đa hoá F2 làm ngưỡng suy biến thành gắn cờ gần như mọi bệnh nhân (Recall 1.0, Specificity ≈ 0.1).
- Mở rộng kiểm toán trạng thái fit cho RobustScaler và PowerTransformer.
- Tập trung quy tắc "mô hình nào cần scale" vào `models.needs_scaling()`.

### 4.5. Giai đoạn 1 — Mở rộng mô hình
- Thêm **Extra Trees** (mô hình tốt nhất trong Ishaq et al. 2021) và **MLP** (hidden (16,), solver lbfgs).
- Sửa lỗi kế thừa từ template: `get_model(..., **overrides)` từng báo TypeError khi tham số override trùng với tham số mặc định. Nay override thay thế giá trị mặc định, và có test kèm theo.

### 4.6. Giai đoạn 2 — Mô hình tối giản 2 biến
- Định nghĩa các tập đặc trưng có tên (`all`, `minimal_two`) trong `config.FEATURE_SETS`.
- Thêm `train.compare_feature_sets()`: chạy **paired grouped CV**, trong đó hai nhánh dùng cùng các fold. Kết quả báo cáo CI 95% (theo phân phối t) cho chênh lệch giữa hai nhánh.
- Thêm tham số CLI `--feature-set {all,minimal_two,compare}`.

### 4.7. Giai đoạn 3 — Đồng bộ dữ liệu Zigong
- Module `src/harmonization.py` ánh xạ cả hai cohort về **9 đặc trưng chung**:
  - tuổi, quy về các nhóm thập niên của Zigong;
  - giới tính, tiểu đường;
  - thiếu máu, tính từ hemoglobin theo ngưỡng WHO (nam < 130 g/L, nữ < 120 g/L);
  - LVEF;
  - creatinine, đổi từ µmol/L sang mg/dL (chia 88,4);
  - natri;
  - tiểu cầu, đổi đơn vị của UCI sang 10⁹/L;
  - CK.
- Gán nhãn tử vong ở các mốc 28, 90 và 180 ngày. Có thêm tuỳ chọn `uci_like` lọc nhóm tuổi ≥ 40 và NYHA III–IV cho giống tiêu chí UCI.
- Script `scripts/inspect_zigong.py` chỉ in thống kê tổng hợp, có chế độ `--synthetic`.
- Có bộ sinh dữ liệu Zigong giả lập để dùng cho test và CI.

### 4.8. Giai đoạn 4 — Kiểm chứng ngoài (phương án A)
- Module `src/external_validation.py` và script `scripts/run_external_validation.py`.
- **Quy trình:**
  1. Huấn luyện trên toàn bộ UCI đã đồng bộ, **không SMOTE**, vì SMOTE làm xác suất dự đoán bị đẩy lên.
  2. Chọn ngưỡng bằng dự đoán out-of-fold từ grouped 5-fold CV trên UCI, với điều kiện Recall ≥ 0.80.
  3. Đánh giá trên Zigong theo 3 nhóm chỉ số:
     - phân biệt: ROC-AUC kèm CI bootstrap phân tầng 1.000 lần, và PR-AUC;
     - hiệu chuẩn: calibration-in-the-large, calibration slope, Brier score, và Brier sau khi hiệu chỉnh intercept;
     - độ nhạy và độ đặc hiệu tại ngưỡng đã chọn trên UCI.
- **Ba tập đặc trưng:**
  - `all9`: 9 biến;
  - `no_ef`: 8 biến, áp dụng cho toàn bộ 2.008 lượt nhập viện;
  - `minimal_two`: creatinine + EF.
  - Các mô hình dùng LVEF chỉ được áp dụng cho 635 lượt nhập viện có đo LVEF.
- **Ba mô hình:** Logistic Regression, Random Forest, Extra Trees.
- **Phân tích độ nhạy:** mốc 28 và 180 ngày, và tập con `uci_like`.
- Thêm lệnh `make inspect-zigong` và `make external-validation`. CI chạy cả hai trên dữ liệu giả lập.

---

## 5. Kết quả

### 5.1. Phân tích khám phá (EDA) trên UCI

| Phân tầng | Tỉ lệ tử vong | Risk ratio |
|---|---|---|
| EF ≤ 40% (HFrEF), n = 219 | 35,2% | 1,00 (tham chiếu) |
| EF ≥ 50% (HFpEF), n = 60 | 23,3% | 0,66 |
| Creatinine ≤ 1,2 mg/dL, n = 198 | 21,2% | 1,00 (tham chiếu) |
| Creatinine ≥ 2,0 mg/dL, n = 35 | 62,9% | 2,96 |
| Natri < 135 mmol/L, n = 83 | 50,6% | 2,02 |
| Tuổi ≥ 80, n = 25 | 72,0% | — |

**Rò rỉ dữ liệu qua `time`:** AUC đơn biến của `time` là 0,84, cao hơn mọi biến lâm sàng; biến lâm sàng mạnh nhất là creatinine với AUC 0,73.

### 5.2. Mô hình trên UCI (11 biến, ngưỡng Recall ≥ 0,80)

Grouped CV trên train+val (trung bình ± độ lệch chuẩn qua 5 fold):

| Mô hình | Recall | Specificity | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0,86 ± 0,07 | 0,49 ± 0,15 | 0,63 ± 0,03 | 0,77 ± 0,04 |
| Random Forest | 0,79 ± 0,15 | 0,68 ± 0,14 | 0,67 ± 0,04 | 0,78 ± 0,05 |
| Gradient Boosting | 0,83 ± 0,13 | 0,46 ± 0,14 | 0,63 ± 0,11 | 0,74 ± 0,06 |
| Stacking Ensemble | 0,78 ± 0,10 | 0,62 ± 0,14 | 0,63 ± 0,11 | 0,76 ± 0,08 |

- Extra Trees đạt ROC-AUC 0,75 và PR-AUC 0,61. MLP yếu hơn, ở mức 0,72 / 0,59.
- **Tập test** (60 bệnh nhân, 19 ca tử vong): Stacking đạt Recall 0,74, Specificity 0,66, ROC-AUC 0,76.
  - Tập test nhỏ nên mỗi bệnh nhân làm Recall thay đổi 0,05. Kết quả CV vì vậy đáng tin hơn.

### 5.3. Mô hình 2 biến so với 11 biến (paired grouped CV)

| Mô hình | ROC-AUC (11 → 2 biến) | PR-AUC (11 → 2 biến) | Δ ROC-AUC [CI 95%] |
|---|---|---|---|
| Logistic Regression | 0,77 → 0,77 | 0,63 → 0,65 | +0,01 [−0,04; 0,05] |
| Random Forest | 0,78 → 0,83 | 0,67 → 0,71 | +0,05 [−0,01; 0,11] |
| Extra Trees | 0,75 → 0,83 | 0,61 → 0,74 | +0,08 [−0,01; 0,16] |
| Gradient Boosting | 0,74 → 0,78 | 0,63 → 0,69 | +0,04 [0,02; 0,06] |

**Kết luận:** mô hình chỉ gồm creatinine và EF **không kém** mô hình 11 biến, khớp với Chicco & Jurman (2020). Tuy vậy, chỉ ở Gradient Boosting thì CI của chênh lệch mới không chứa 0.

### 5.4. Giải thích mô hình
- SHAP trên Random Forest xếp `serum_creatinine` và `ejection_fraction` ở hai vị trí đầu.
- Odds ratio của Logistic Regression, tính theo mỗi 1 độ lệch chuẩn:
  - creatinine: OR 3,7 (CI 95%: 2,1–6,6);
  - EF: OR 0,43 (0,28–0,67), tức EF cao có tác dụng bảo vệ.
- Các CI này chỉ mang tính tham khảo, vì mô hình học trên dữ liệu đã qua SMOTE.

### 5.5. Kiểm chứng ngoài trên Zigong — phân tích chính (tử vong 90 ngày)

| Mô hình (đặc trưng) | Zigong n / tử vong | AUC nội bộ UCI | ROC-AUC Zigong [CI 95%] | Nguy cơ dự đoán TB / thực tế | CITL | Slope | Sens / Spec tại ngưỡng UCI |
|---|---|---|---|---|---|---|---|
| LR (no_ef, 8 biến) | 2.008 / 42 | 0,66 | 0,65 [0,56; 0,74] | 0,34 / 0,021 | −3,37 | 0,83 | 0,95 / 0,18 |
| RF (no_ef) | 2.008 / 42 | 0,64 | 0,61 [0,52; 0,70] | 0,37 / 0,021 | −3,47 | 0,53 | 0,93 / 0,11 |
| ET (no_ef) | 2.008 / 42 | 0,66 | 0,59 [0,50; 0,69] | 0,34 / 0,021 | −3,25 | 0,68 | 0,90 / 0,13 |
| LR (all9) | 635 / 10 | 0,68 | 0,67 [0,48; 0,83] | 0,24 / 0,016 | −3,19 | 0,61 | 0,80 / 0,35 |
| RF (all9) | 635 / 10 | 0,71 | 0,68 [0,51; 0,82] | 0,31 / 0,016 | −3,56 | 0,76 | 0,90 / 0,33 |
| LR (creatinine + EF) | 635 / 10 | 0,67 | 0,72 [0,58; 0,86] | 0,17 / 0,016 | −2,66 | 1,05 | 0,60 / 0,74 |
| RF (creatinine + EF) | 635 / 10 | 0,73 | 0,79 [0,71; 0,87] | 0,18 / 0,016 | −2,89 | 0,82 | 0,60 / 0,69 |
| ET (creatinine + EF) | 635 / 10 | 0,73 | 0,77 [0,63; 0,88] | 0,23 / 0,016 | −2,95 | 2,32 | 0,70 / 0,68 |

*CITL = calibration-in-the-large, tính trên thang logit; giá trị 0 nghĩa là hiệu chuẩn hoàn hảo.*

**Phân tích độ nhạy:** kết luận không đổi ở mọi kịch bản.

| Kịch bản | ROC-AUC LR no_ef | ROC-AUC RF minimal_two | CITL |
|---|---|---|---|
| 28 ngày | 0,63 | 0,76 | −1,6 đến −3,0 |
| 180 ngày | 0,66 | 0,76 | −3,5 đến −4,4 |
| Tập con giống UCI (90 ngày) | 0,64 | 0,78 | −2,6 đến −3,5 |

### 5.6. Diễn giải
1. **Khả năng phân biệt chuyển giao ở mức khiêm tốn.** AUC trên Zigong gần với AUC nội bộ của chính mô hình trên UCI ở cùng mốc 90 ngày (0,64–0,73). Như vậy mô hình không bị sụp đổ khi sang Zigong, nhưng bản thân nó vốn đã yếu.
2. **Mô hình càng đơn giản càng chuyển giao tốt.**
   - Khi không có LVEF, Logistic Regression vượt các mô hình cây.
   - Mô hình 2 biến cho AUC cao nhất. Tuy nhiên nhóm có LVEF chỉ có 10 ca tử vong, nên CI rất rộng và không đủ để kết luận mạnh.
3. **Hiệu chuẩn thất bại.**
   - Nguyên nhân: UCI có tỉ lệ tử vong 27% ở mốc 90 ngày, trong khi Zigong chỉ có 2%.
   - Mô hình vì vậy dự đoán nguy cơ cao gấp khoảng 16 lần thực tế.
   - Hiệu chỉnh intercept chỉ đưa Brier score về mức của một mô hình không có kỹ năng.
4. **Ngưỡng chọn trên UCI không dùng được trên Zigong.** Tại ngưỡng này mô hình gắn cờ gần như toàn bộ bệnh nhân (độ đặc hiệu 0,11–0,18).
5. Kết quả khớp với nhận định trong tài liệu: mô hình ML phát triển trên một cohort nhỏ cần được hiệu chỉnh lại tại chỗ trước khi triển khai ở nơi khác.

---

## 6. Hạn chế
- UCI nhỏ (299 bệnh nhân) và chỉ đến từ một trung tâm ở Faisalabad năm 2015. Tập test chỉ có 19 ca tử vong.
- Zigong không có ID bệnh nhân, nên không gộp được các lượt nhập viện lặp lại (7%).
- Thiếu hai biến tăng huyết áp và hút thuốc. LVEF thiếu 68% và việc thiếu không ngẫu nhiên.
- Đơn vị CK của Zigong chưa được xác nhận. Tuổi ở Zigong chỉ có dạng nhóm thập niên.
- Hai cohort khác biệt lớn về bối cảnh: bệnh nhân UCI là suy tim nặng (NYHA III/IV, theo dõi ngoại trú), còn Zigong là các lượt nhập viện.
- Hệ số Logistic Regression học trên dữ liệu đã qua SMOTE, nên odds ratio chỉ mang tính tham khảo.

---

## 7. Cấu trúc mã nguồn và kiểm thử

| Thành phần | Nội dung |
|---|---|
| `src/` | `config`, `data_loader`, `preprocessing`, `models`, `evaluation`, `explainability`, `eda`, `train`, `harmonization`, `external_validation` |
| `scripts/` | `download_data`, `run_eda`, `inspect_zigong`, `run_external_validation` |
| `tests/` | 172 test, gồm bộ chống rò rỉ `test_data_leakage.py`, `test_feature_sets.py`, `test_harmonization.py`, `test_external_validation.py` |
| `notebooks/` | 01 EDA, 02 tiền xử lý và mất cân bằng, 03 huấn luyện và đánh giá (đã chạy sẵn) |
| `reports/` | Bảng so sánh mô hình, kết quả CV, EDA, SHAP. Kết quả Zigong nằm trong `reports/external_validation/` và bị git bỏ qua |
| CI | Lint, test, EDA, đồng bộ dữ liệu và kiểm chứng ngoài, tất cả chạy trên dữ liệu giả lập |

**Lệnh chính:**
- `make verify`, `make test`, `make eda`;
- `make train`, `make train-recall80`, `make compare-features`;
- `make inspect-zigong`, `make external-validation`.

**Lịch sử commit:**

| Commit | Nội dung | Trạng thái |
|---|---|---|
| `405fcd0` | Khởi tạo dự án | Đã push |
| `d5a76c0` | Tài liệu nghiên cứu của nhóm | Đã push |
| `a222f91` | Kế hoạch v1.1 | Đã push |
| `3e1dc34` | Giai đoạn 0 | Đã push |
| `90f547f` | Giai đoạn 1 | Đã push |
| `0ab9402` | Giai đoạn 2 | Đã push |
| `ea504fc` | Giai đoạn 3 | Chưa push |
| `dccfbde` | Giai đoạn 4 | Chưa push |

---

## 8. Việc tiếp theo

| Ưu tiên | Việc | Người thực hiện |
|---|---|---|
| Cao | Push `ea504fc` và `dccfbde` lên GitHub | Chờ nhóm đồng ý |
| Cao | Tải lại từ điển biến Zigong để xác nhận đơn vị CK | Nhóm, qua tài khoản PhysioNet |
| Trung bình | **Giai đoạn 5:** biểu đồ hiệu chuẩn (UCI có/không SMOTE, Zigong trước/sau hiệu chỉnh), đường PR, SHAP ở 300 DPI; `references.bib` kèm DOI | Claude / Codex |
| Thấp | **Giai đoạn 6:** tuỳ chọn RobustScaler, mặc định vẫn là StandardScaler | Claude / Codex |
| Tuỳ chọn | **Giai đoạn 7:** phân tích sống còn bằng Cox, trong đó `time` là biến kết cục chứ không phải đặc trưng | Claude / Codex |
| Đề xuất | Hiệu chỉnh lại mô hình bằng một phần dữ liệu Zigong (temporal split), rồi đánh giá trên phần còn lại | Cần nhóm quyết định |
