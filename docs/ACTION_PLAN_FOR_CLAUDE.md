# KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT (ACTION PLAN) CHO CLAUDE CODE / CODEX
## Dự án: Heart Failure Mortality Prediction using Big Data & Machine Learning
**Đường dẫn thư mục dự án:** `D:\Antigravity\heart-failure`
**Repo:** https://github.com/fossbk-spec/heart-failure
**Phiên bản:** v1.3 (2026-10-14). v1.1 (2026-10-13) đã sửa v1.0 sau khi đối chiếu với mã nguồn; v1.2 bổ sung các Giai đoạn 8–11 theo kết quả Codex review và phân tích gộp UCI + Zigong; v1.3 thêm Giai đoạn 12 (bộ dữ liệu kiểm chứng ngoài thứ hai và thứ ba).
**Hướng dẫn cho sinh viên:** [`STUDENT_GUIDE_v1.2.md`](STUDENT_GUIDE_v1.2.md)

---

## 0b. THAY ĐỔI CỦA v1.3 SO VỚI v1.2

| # | v1.2 | v1.3 | Lý do |
|---|---|---|---|
| 1 | Chỉ có 2 bộ dữ liệu (UCI, Zigong) | Giai đoạn 12: thêm MIMIC-III (ưu tiên 1) và TOPCAT / SCD-HeFT qua BioLINCC (ưu tiên 2) | Nhóm Zigong có LVEF chỉ có 10 ca tử vong; cần một bộ kiểm chứng ngoài có LVEF và nhiều ca tử vong hơn |
| 2 | Giai đoạn 7 (Cox) hoãn | Có thể mở lại khi có dữ liệu thử nghiệm lâm sàng theo dõi dài (TOPCAT, SCD-HeFT) | Các thử nghiệm này có thời gian đến tử vong được thẩm định |

**Đã loại sau khi khảo sát (2026-10-14):** Kaggle "Heart Failure Prediction" (918 dòng, là bài toán chẩn đoán bệnh tim, không có kết cục tử vong); các bản sao UCI trên figshare/Kaggle (trùng bệnh nhân); Dryad "CHF in ICU" (Han et al., 15,983 BN; không có EF, natri, tiểu cầu, thiếu máu, tiểu đường, giới); bản DIG "teaching" (đã xáo trộn để ẩn danh, không dùng để công bố).

## 0a. THAY ĐỔI CỦA v1.2 SO VỚI v1.1

| # | v1.1 | v1.2 | Lý do |
|---|---|---|---|
| 1 | Giai đoạn 2 kết luận "2 biến không kém 11 biến" | Chỉ kết luận "điểm ước lượng tương đương hoặc cao hơn" | Codex review: CI t trên 5 fold phụ thuộc nhau quá hẹp; CI chứa 0 không chứng minh được non-inferiority |
| 2 | Giai đoạn 4 mục 3: "Cohort B: bỏ LVEF, thêm biến chỉ báo thiếu" | Cohort B chỉ bỏ LVEF; biến chỉ báo thiếu được đánh giá riêng ở Giai đoạn 4b | Mã nguồn không thêm biến chỉ báo; Giai đoạn 4b cho thấy biến này không giúp |
| 3 | (chưa có) | Giai đoạn 4b: gộp UCI + Zigong, so sánh 7 cách xử lý LVEF thiếu | Yêu cầu của nhóm (2026-10-14) |
| 4 | Giai đoạn 5: biểu đồ + `references.bib` | Phần lớn đã làm trong bản thảo paper (`docs/paper/`); phần còn lại chuyển vào Giai đoạn 10 | Paper IEEE đã có Hình 1–2 và 27 tài liệu tham khảo |
| 5 | Giai đoạn 6 (RobustScaler), 7 (Cox) | **Hoãn** | Không ảnh hưởng các kết luận hiện tại |
| 6 | (chưa có) | Giai đoạn 8–11: sửa theo Codex review → bổ sung phân tích → cập nhật paper → Codex review lại | Codex review 2026-10-14: 0 nghiêm trọng, 3 lớn, 7 nhỏ (`docs/CODEX_REVIEW.md`) |
| 7 | Hiệu chỉnh trên Zigong theo thời gian | Chia ngẫu nhiên phân tầng, lặp lại | Zigong không có ngày nhập viện (chỉ có `dischargeDay`, `visit.times`) |

## 0. THAY ĐỔI CỦA v1.1 SO VỚI v1.0

| # | v1.0 | v1.1 | Lý do |
|---|---|---|---|
| 1 | Hàm `build_preprocessing_pipeline` | `make_preprocessor` / `make_full_pipeline` (`src/preprocessing.py`) | Hàm cũ không tồn tại |
| 2 | Test trong `tests/test_train.py` | `tests/test_models.py`, `tests/test_feature_sets.py` | File cũ không tồn tại |
| 3 | "12 đặc trưng" | 11 đặc trưng | `time` bị loại (rò rỉ thời gian) |
| 4 | Thêm `extra_trees` vào `builders` là đủ | + `available_models()` + quy tắc không scale cho mô hình cây | Nếu không, CLI và pipeline xử lý sai |
| 5 | MLP `early_stopping=True` | Không early stopping, mạng nhỏ + `alpha` | ~190 BN train → ~19 mẫu validation nội bộ, lẫn mẫu SMOTE |
| 6 | Mục tiêu ETC 92.62% (Ishaq 2021) | Không đặt mục tiêu; kỳ vọng ETC ≈ RF | 92.62% là Accuracy, nghi SMOTE rò rỉ |
| 7 | So sánh 2 biến vs đầy đủ trên test | So sánh **cặp** trên cùng các fold grouped CV + CI của hiệu số | Test chỉ có 19 ca tử vong |
| 8 | (thiếu) MCC | Thêm MCC vào `compute_metrics` | Tài liệu 03 coi MCC là chỉ số chính |
| 9 | RobustScaler làm mặc định | Tùy chọn, mặc định giữ StandardScaler; thêm `center_` vào bộ kiểm tra rò rỉ | Mặc định robust làm gãy test hiện có |
| 10 | Kết cục Zigong = `DestinationDischarge` (tử vong nội viện) | **Tử vong trong 90 ngày** (chính) + phân tích sống còn (độ nhạy) | Chỉ 14/2008 ca tử vong nội viện (Zhang 2021) |
| 11 | Đồng bộ dữ liệu trong `data_loader.py` | Module riêng `src/harmonization.py` | `validate_schema` ràng buộc theo schema UCI |
| 12 | Calibration với pipeline SMOTE | Thêm nhánh không SMOTE (`class_weight="balanced"`), báo cáo calibration slope/intercept | SMOTE làm lệch xác suất |
| 13 | `references.bib` 8 mục (liệt kê 7) | Gộp `docs/literature/` (7 bài, đã xác minh PubMed) + `01_Literature_Review.md`, có DOI; bỏ slide HUST khỏi trích dẫn | Hai bộ tài liệu song song, slide không phải nguồn trích dẫn được |

---

## 1. NGUYÊN TẮC KỸ THUẬT BẤT BIẾN

1. **Không làm gãy test hiện có.** Chạy sau mỗi thay đổi:
   ```powershell
   & "D:\Antigravity\heart-failure\.venv\Scripts\python.exe" -m pytest
   & "D:\Antigravity\heart-failure\.venv\Scripts\python.exe" -m ruff check src tests scripts
   ```
2. **Zero Data Leakage:**
   - Chia theo bệnh nhân (`patient_level_split` / `StratifiedGroupKFold`) luôn diễn ra **đầu tiên**.
   - Imputer, log1p, Scaler, SMOTE chỉ `fit` trên Train; Validation/Test chỉ `.transform()`.
   - SMOTE chỉ nằm trong `imblearn.pipeline.Pipeline` (và trong từng base learner của Stacking).
   - `time` không bao giờ là đặc trưng đầu vào. Chỉ được dùng làm **thời gian kết cục** (Giai đoạn 7) hoặc để tạo nhãn theo mốc cố định (Giai đoạn 3).
   - Chọn đặc trưng / tối ưu siêu tham số nếu có phải nằm **bên trong** vòng CV (nested CV có gom nhóm).
3. **Dữ liệu Zigong (PhysioNet Restricted Health Data License 1.5.0 + DUA):**
   - Chỉ đặt tại `data/raw/zigong/` (đã được `.gitignore` chặn). **Không commit, không đưa lên CI, không tải lên Colab/Kaggle public.**
   - Test dùng dữ liệu giả lập cùng cấu trúc Zigong.
   - Nhóm đã có tài khoản PhysioNet xác thực và đã ký DUA; nhóm cho phép trợ lý AI đọc dữ liệu trong phạm vi dự án.
4. **Môi trường:** `D:\Antigravity\heart-failure\.venv\Scripts\python.exe`.
5. **Git:** mỗi giai đoạn một commit; CI phải xanh. Không thêm dòng đồng tác giả AI vào commit/PR.

---

## 2. LỘ TRÌNH TRIỂN KHAI

### GIAI ĐOẠN 0: SỬA NỀN (không cần Zigong)
- [x] Thêm `mcc` (Matthews Correlation Coefficient) vào `src/evaluation.compute_metrics`; in trong bảng của `src/train.py`.
- [x] `_fitted_state` (`src/preprocessing.py`) theo dõi thêm `center_` (RobustScaler) và `lambdas_` (PowerTransformer).
- [x] Gom quy tắc "mô hình cây không cần scale" về một chỗ: `models.needs_scaling(name)`; `train.py` dùng hàm này.
- [x] Test: MCC tính tay; phát hiện fit lại RobustScaler trên hold-out.

### GIAI ĐOẠN 1: MỞ RỘNG MÔ HÌNH (không cần Zigong)
- [x] `extra_trees`: `ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight=..., n_jobs=-1)` — cùng mặc định với Random Forest; không scale.
- [x] `mlp`: `MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, solver="lbfgs", max_iter=2000)`, không early stopping; có scale (`adam` không hội tụ trong 2000 vòng). MLP không hỗ trợ `class_weight` → dùng SMOTE.
- [x] Thêm cả hai vào `available_models()`; thêm `extra_trees` vào `DEFAULT_MODELS`.
- [x] XGBoost: giữ tùy chọn (`requirements.txt` để ở dạng chú thích), không bắt buộc.
- [x] Test `tests/test_models.py`: khởi tạo, fit/predict trong pipeline, quy tắc scale, `class_weight`.
- [x] Sửa lỗi kế thừa từ template: `get_model(..., n_estimators=50)` gây `TypeError` (tham số trùng) → nay `overrides` ghi đè được giá trị mặc định.

### GIAI ĐOẠN 2: MÔ HÌNH TỐI GIẢN 2 BIẾN — CHICCO & JURMAN 2020 (không cần Zigong)
- [x] `config.FEATURE_SETS = {"all": 11 đặc trưng, "minimal_two": ["ejection_fraction", "serum_creatinine"]}`.
- [x] `split_features_target(..., feature_set=)`, `make_preprocessor(..., feature_set=)`, `make_full_pipeline(..., feature_set=)`; chỉ số cột log1p tính theo tập đặc trưng.
- [x] `src/train.py`: `--feature-set {all, minimal_two, compare}`.
  - `compare`: grouped CV trên train+val, **cùng các fold** cho cả hai tập đặc trưng → hiệu số theo cặp (minimal − all), trung bình, độ lệch chuẩn, CI 95% (t, df = k−1; xấp xỉ vì các fold không độc lập).
  - Ghi `reports/minimal_vs_full_feature_comparison.csv`: Recall, Specificity, PR-AUC, ROC-AUC, MCC, Brier, F2.
- [x] Test `tests/test_feature_sets.py`.
- **Kết quả (paired grouped CV 5 fold trên train+val, Recall ≥ 0.80):** mô hình 2 biến có điểm ước lượng tương đương hoặc cao hơn mô hình 11 biến (v1.2: không kết luận "không kém", xem Giai đoạn 8.2); ROC-AUC RF 0.78 → 0.83, Extra Trees 0.75 → 0.83, GB 0.74 → 0.78 (CI hiệu số chỉ loại trừ 0 ở GB). Ủng hộ Chicco & Jurman (2020) về hướng, chưa đủ mạnh về thống kê với 239 BN.

### GIAI ĐOẠN 3: TIẾP NHẬN & ĐỒNG BỘ DỮ LIỆU ZIGONG (cần dữ liệu)
**Kết cục đã chọn (so sánh các phương án):**

| Phương án | Zigong | UCI giữ lại | Ca tử vong UCI | Kết luận |
|---|---|---|---|---|
| A. Tử vong nội viện | 14 ca (0.7%) | Không ghép được | — | ❌ |
| B. Tử vong ≤ 28 ngày | Có | 296/299 | 30 (10.1%) | ⚠️ Quá ít ca |
| **C. Tử vong ≤ 90 ngày** | Có (3 tháng) | 256/299 (loại 43 BN kiểm duyệt sớm) | 69 (27.0%) | ✅ **Kết cục chính** |
| D. Tử vong ≤ 180 ngày | Có (6 tháng) | 193/299 (loại 106) | 89 (46.1%) | ⚠️ Thiên lệch chọn mẫu |
| E. Phân tích sống còn | Thời gian đến tử vong, kiểm duyệt 6 tháng | 299/299 | 96 | ✅ Độ nhạy (Giai đoạn 7) |
| F. Tử vong/tái nhập viện 6 tháng | Có | Không có biến | — | ❌ Chỉ dùng nội bộ Zigong |

Quy tắc nhãn UCI ở mốc H ngày: `1` nếu `DEATH_EVENT=1` và `time ≤ H`; `0` nếu còn sống tại H (`time ≥ H`, hoặc tử vong sau H); **loại** nếu còn sống nhưng bị kiểm duyệt trước H. Pipeline UCI chính vẫn giữ nhãn `DEATH_EVENT` để so sánh được với y văn.

- [x] Nhóm đặt dữ liệu Zigong v1.3 vào `data/raw/zigong/` (`dat.csv`, `dat_md.csv`; file mô tả biến chưa tải xong — có file `.crdownload` 0 byte).
- [x] `scripts/inspect_zigong.py` (chỉ in số liệu tổng hợp): tên cột, kiểu, đơn vị, tỷ lệ thiếu, số ca từng kết cục, biến định danh bệnh nhân, số lần nhập viện/BN.
- [x] Chốt bảng ánh xạ (suy từ phân bố giá trị; cần đối chiếu lại file mô tả biến khi tải được). Các điểm đã kiểm tra:
  - Đơn vị creatinine (µmol/L → mg/dL: chia 88.4), tiểu cầu (UCI ~263,358/µL ↔ 10⁹/L: chia 1000), CK (U/L vs mcg/L).
  - Tuổi: nếu Zigong ghi theo nhóm thập kỷ thì UCI chia nhóm giống hệt.
  - `smoking`: không có ở Zigong → bỏ khỏi tập đặc trưng chung.
  - `anaemia`: suy ra từ hemoglobin (WHO: nam < 13 g/dL, nữ < 12 g/dL).
  - Tăng huyết áp: chưa thấy trong danh sách bệnh đồng mắc của Zhang 2021 → xác minh.
  - **Biến định danh bệnh nhân**: 7% có ≥ 2 lần nhập viện (`visit.times`); nếu không gom nhóm được → ghi rõ là hạn chế.
- [x] `src/harmonization.py`: đổi tên biến, quy đổi đơn vị, suy ra `anaemia`, nhãn 90 ngày cho cả hai bộ, cột `source`; tập đặc trưng chung.
- [x] Test bằng dữ liệu giả lập cùng cấu trúc Zigong (`tests/test_harmonization.py`, CI không cần dữ liệu thật).

**Kết quả kiểm tra dữ liệu Zigong thật (2026-10-13):**

| Hạng mục | Kết quả |
|---|---|
| Kích thước | 2,008 dòng × 167 cột; mỗi dòng = 1 lần nhập viện (`inpatient.number` duy nhất) |
| Định danh bệnh nhân | **Không có**; `visit.times` ≥ 2 ở 148 dòng (7%) → không gom nhóm được, ghi là hạn chế |
| Số ca tử vong | 28 ngày: 37 (1.8%) · **90 ngày: 42 (2.1%)** · 180 ngày: 57 (2.8%) — tất cả < ~100 |
| UCI theo mốc | 28 ngày: 30/296 · 90 ngày: 69/256 · 180 ngày: 89/193 |
| Tuổi | Chỉ có `ageCat` 8 nhóm (21–110); UCI được chia nhóm giống hệt (giá trị đại diện 25…95) |
| Creatinine | `creatinine.enzymatic.method` µmol/L (trung vị 87.1) → /88.4 = 0.99 mg/dL (UCI 1.1) ✓ |
| Tiểu cầu | `platelet` 10⁹/L (trung vị 135); UCI /1000 → 259.5 |
| CK | `creatine.kinase` trung vị 91 vs UCI 259 — giả định cùng U/L, **cần xác minh** |
| Natri | `sodium` mmol/L (trung vị 139 vs 136) ✓ |
| Thiếu máu | Suy ra từ `hemoglobin` (g/L, WHO) → 60.9% (UCI 44.1%) |
| Không có | Tăng huyết áp, hút thuốc → bỏ khỏi tập đặc trưng chung |
| LVEF | Thiếu 68.4%; có LVEF ở 31.8% BN sống vs 23.8% BN tử vong 3 tháng → thiếu không ngẫu nhiên |
| Khác biệt quần thể | Zigong: trung vị tuổi 75, nữ 57.9%, EF trung vị 51 (HFpEF nhiều); UCI: 65, 35.9%, EF 38 |
| Tiêu chí giống UCI (`uci_like`) | Tuổi ≥ 40 + NYHA III–IV → 1,643 dòng, 37 ca tử vong 90 ngày |
| Chất lượng nhãn | 5 dòng `DestinationDischarge = Died` nhưng `death.within.3.months = 0` → giữ nguyên, báo cáo |

**Hệ quả cho Giai đoạn 4:** với 42 ca tử vong 90 ngày, kiểm chứng ngoài UCI → Zigong chỉ cho ước lượng phân biệt (AUC) có CI rộng; tỷ lệ tử vong chênh 13 lần (27% vs 2.1%) nên bắt buộc hiệu chỉnh lại intercept. Huấn luyện trên Zigong (42 ca / 9 đặc trưng ≈ 4.7 ca/đặc trưng) không đủ cho mô hình ML phức tạp.

### GIAI ĐOẠN 4: KIỂM CHỨNG NGOÀI (cần dữ liệu)
- [x] `scripts/run_external_validation.py` + `src/external_validation.py` (phương án A — chỉ chiều UCI → Zigong):
  1. Huấn luyện trên UCI (tập đặc trưng chung, nhãn 90 ngày, không SMOTE) → đánh giá trên Zigong. Chiều ngược lại bị loại (Zigong chỉ có 42 ca tử vong).
  2. ~~Mô hình nội bộ Zigong~~ — bỏ theo phương án A (không đủ ca tử vong, không có ID bệnh nhân).
  3. **Cohort A**: chỉ BN có LVEF (thiếu không ngẫu nhiên → báo cáo thiên lệch chọn mẫu). **Cohort B**: bỏ LVEF; không nội suy 68% LVEF. (v1.2: biến chỉ báo thiếu được đánh giá ở Giai đoạn 4b.)
  4. Báo cáo khác biệt quần thể (Zigong: tuổi trung vị 77, 58% nữ; UCI: tuổi TB 61, 35% nữ) và lệch hiệu chuẩn.
- [x] Số ca tử vong 90 ngày ở Zigong = 42 (< ~100): báo cáo CI bootstrap, không kết luận mạnh.

**Kết quả (2026-10-13, phân tích chính: tử vong 90 ngày, toàn bộ Zigong):**

| Mô hình (đặc trưng) | Zigong n / tử vong | ROC-AUC [CI 95%] | Nguy cơ dự đoán TB vs quan sát | Calibration slope |
|---|---|---|---|---|
| Logistic Regression (8, không LVEF) | 2,008 / 42 | 0.65 [0.56, 0.74] | 0.34 vs 0.021 | 0.83 |
| Random Forest (8, không LVEF) | 2,008 / 42 | 0.61 [0.52, 0.70] | 0.37 vs 0.021 | 0.53 |
| Extra Trees (8, không LVEF) | 2,008 / 42 | 0.59 [0.50, 0.69] | 0.34 vs 0.021 | 0.68 |
| Logistic Regression (creatinine + EF) | 635 / 10 | 0.72 [0.58, 0.86] | 0.17 vs 0.016 | 1.05 |
| Random Forest (creatinine + EF) | 635 / 10 | 0.79 [0.71, 0.87] | 0.18 vs 0.016 | 0.82 |

- Khả năng phân biệt chuyển giao ở mức khiêm tốn; mô hình càng đơn giản càng chuyển giao tốt (LR > cây khi không có LVEF; mô hình 2 biến tốt nhất nhưng chỉ có 10 ca tử vong).
- Hiệu chuẩn thất bại: dự đoán gấp ~16 lần nguy cơ thực (calibration-in-the-large ≈ −3.4). Sau khi hiệu chỉnh intercept tại chỗ, Brier chỉ bằng mức không có kỹ năng → không được dùng trên quần thể mới khi chưa hiệu chỉnh lại.
- Kết quả nhất quán ở mốc 28 / 180 ngày và tập con giống UCI.

### GIAI ĐOẠN 4b: GỘP UCI + ZIGONG, XỬ LÝ LVEF BỊ THIẾU (đã xong, 2026-10-14)
- [x] `src/pooled_missing.py`, `scripts/run_pooled_missing.py`, `make pooled-missing`, `tests/test_pooled_missing.py` (16 test, tổng 188).
- [x] 7 kịch bản: complete case (bỏ dòng thiếu LVEF, 891 dòng); bỏ cột LVEF (2,264 dòng); điền median / MICE / KNN; median và MICE + biến chỉ báo thiếu. Mỗi kịch bản chạy với và không có biến cohort.
- [x] CV phân tầng 5 fold × 3 lần lặp (phân tầng theo cohort × kết cục); tiền xử lý chỉ fit trên fold huấn luyện; log1p chọn cột theo **tên**.
- [x] Đo độ chính xác khi điền LVEF: che LVEF đã đo ở Zigong, điền lại, so với giá trị thật.

**Kết quả (tử vong 90 ngày, không có biến cohort; khoảng giá trị qua LR / RF / ET):**

| Kịch bản | Số dòng | AUC trong Zigong (tất cả / có LVEF) | AUC trong UCI |
|---|---|---|---|
| Bỏ dòng thiếu LVEF | 891 | – / 0.74–0.82 | 0.62–0.65 |
| Bỏ cột LVEF | 2,264 | 0.72–0.76 / 0.77–0.84 | 0.51–0.55 |
| Điền median / MICE / KNN | 2,264 | 0.70–0.75 / 0.76–0.81 | 0.61–0.64 |
| Điền + biến chỉ báo thiếu | 2,264 | 0.64–0.71 / 0.78–0.80 | 0.62–0.64 |

- AUC trên dữ liệu gộp (0.80–0.89) bị thổi phồng vì hai cohort chênh 13 lần về tỷ lệ tử vong → chỉ so sánh AUC trong từng cohort.
- Điền LVEF: MAE 10.8–11.9 điểm EF, tương quan với giá trị thật r ≈ 0.2, độ phân tán bằng 34–44% thực tế → gần như bịa dữ liệu, không cải thiện AUC.
- LVEF quan trọng với UCI (bỏ LVEF: AUC trong UCI 0.63 → 0.53). Biến chỉ báo thiếu không giúp (tử vong 2.3% không đo vs 1.6% có đo).
- Hiệu chuẩn trên Zigong: CITL −0.4 đến −1.3 (−0.05 đến −0.5 khi có biến cohort) — là kiểm chứng nội bộ, không phải khả năng chuyển giao.
- **Khuyến nghị:** bỏ cột LVEF cho mô hình áp dụng mọi lượt nhập viện; complete case khi cần LVEF; điền giá trị chỉ là phân tích độ nhạy.

### GIAI ĐOẠN 5: BIỂU ĐỒ & TRÍCH DẪN (v1.2: phần lớn đã làm trong `docs/paper/`, phần còn lại chuyển sang Giai đoạn 10)
- [ ] `scripts/generate_clinical_figures.py` (300 DPI, `reports/figures/`): calibration curve (LR / RF / ETC, có và không SMOTE) + calibration slope/intercept + Brier; đường PR; SHAP summary & bar.
- [ ] `data/SSoT_pdf_references/references.bib`: gộp 2 bộ tài liệu, có DOI. Xác minh với PDF:
  - Sun 2022: 202 + 78 là số **nghiên cứu**; con số "97% High Risk of Bias" không có trong abstract.
  - Zhang 2021: abstract ghi 166 thuộc tính, PhysioNet ghi 168.
  - Kapelios 2023: xác minh tên tạp chí.

### GIAI ĐOẠN 6: ROBUSTSCALER (ưu tiên thấp — v1.2: HOÃN)
- [ ] `make_preprocessor(scaler="standard" | "robust")`, mặc định `standard`. Lợi ích nhỏ vì đã có log1p.

### GIAI ĐOẠN 7: PHÂN TÍCH SỐNG CÒN (tùy chọn, độ nhạy — v1.2: HOÃN)
- [ ] Cox PH (lifelines hoặc scikit-survival), C-index, AUC phụ thuộc thời gian tại 90 ngày; `time` là thời gian kết cục.

### GIAI ĐOẠN 8: SỬA THEO CODEX REVIEW (≈ 1 ngày; chỉ bước nghiệm thu cần Zigong)
Nguồn: `docs/CODEX_REVIEW.md` (2026-10-14).

| # | Việc | Codex | Ưu tiên | File chính |
|---|---|---|---|---|
| 8.1 | Thay Newton thuần trong `_fit_logistic` bằng cách giải ổn định: intercept-only dùng tìm nghiệm có chặn (`scipy.optimize.brentq`); slope dùng Newton có line search hoặc `scipy.optimize.minimize`; trả về trạng thái hội tụ; phát hiện tách hoàn toàn (separation) | Lớn #3 | Cao | `src/external_validation.py` |
| 8.2 | So sánh 2 biến vs 11 biến: CV lặp 10×5 + kiểm định t hiệu chỉnh Nadeau–Bengio (hoặc bootstrap theo bệnh nhân lặp lại toàn bộ quy trình fit); bỏ CI t thường; diễn đạt mô tả | Lớn #2 | Cao | `src/train.py` |
| 8.3 | Stacking: truyền danh sách fold nội bộ chia theo bệnh nhân vào `StackingClassifier(cv=...)`, hoặc báo lỗi khi `patient_id` trùng lặp | Lớn #1 | Trung bình | `src/models.py`, `src/train.py` |
| 8.4 | `make_preprocessor`: chọn cột log1p theo tên (như `pooled_missing`); `validate_model`: cohort áp dụng rỗng → trả NaN, không lỗi | Nhỏ #4, #5 | Trung bình | `src/preprocessing.py`, `src/external_validation.py` |
| 8.5 | Giới hạn `--models` của kiểm chứng ngoài về LR/RF/ET; test spy dùng index hai cohort không trùng nhau và kiểm tra cả dữ liệu chọn ngưỡng; sửa `KeyError: 'f2'` khi `--beta` ≠ 2 | Nhỏ #6, #7, #8 | Trung bình | `scripts/run_external_validation.py`, `tests/test_external_validation.py`, `src/train.py` |
| 8.6 | CI odds ratio: ma trận thông tin có cả intercept; README/báo cáo mô tả đúng phạm vi của `assert_no_fit_on_holdout` và `verify_zero_data_leakage` | Nhỏ #9, #10 | Thấp | `src/explainability.py`, `README.md` |

**Nghiệm thu Giai đoạn 8:**
- [ ] Mỗi mục có ít nhất một test tái hiện đúng kịch bản lỗi Codex mô tả (fail trước khi sửa, pass sau khi sửa).
- [ ] Chạy lại `make train-recall80`, `make compare-features`, `make external-validation`, `make pooled-missing`; lập bảng **số cũ → số mới** cho mọi con số có trong README và paper.
- [ ] Toàn bộ test pass; `ruff check src tests scripts` sạch.

### GIAI ĐOẠN 9: BỔ SUNG PHÂN TÍCH (≈ 1–2 ngày, cần Zigong)
| # | Việc | Ưu tiên |
|---|---|---|
| 9.1 | **Hiệu chỉnh trên một phần Zigong, đánh giá trên phần giữ lại.** Chia ngẫu nhiên phân tầng 50/50, lặp ≥ 200 lần (Zigong không có ngày nhập viện). So sánh: (a) không hiệu chỉnh, (b) chỉ intercept, (c) intercept + slope (logistic recalibration), (d) huấn luyện lại trên UCI + phần Zigong. Báo cáo AUC, CITL, slope, Brier trên phần giữ lại (trung vị và khoảng 2.5–97.5%) | Cao |
| 9.2 | Phân tích gộp: CV lặp 10×5; CI bootstrap cho AUC trong từng cohort | Cao |
| 9.3 | Độ nhạy: chỉ giữ lượt nhập viện đầu tiên (`visit.times == 1`) cho kiểm chứng ngoài và phân tích gộp | Trung bình |
| 9.4 | (Tuỳ chọn) Huấn luyện trên Zigong (8 biến, không LVEF) → kiểm chứng trên UCI; chỉ mang tính tham khảo (42 ca tử vong) | Thấp |
| 9.5 | (Tuỳ chọn) Điền LVEF theo giả định thiếu không ngẫu nhiên: dịch giá trị điền ±5 / ±10 điểm EF (delta adjustment) | Thấp |

**Nghiệm thu:** module + script + test trên dữ liệu giả lập; kết quả chỉ ở dạng tổng hợp trong `reports/` (git-ignored); cập nhật README.

### GIAI ĐOẠN 10: CẬP NHẬT PAPER (≈ 1 ngày)
- [x] Bản nháp mục "Pooled Analysis and Missing LVEF" (Methods + Results, Table V) với kết quả Giai đoạn 4b — đã thêm ngày 2026-10-14.
- [ ] Kết quả 9.1 → Discussion chuyển từ "cần hiệu chỉnh" thành "hiệu chỉnh cải thiện được bao nhiêu".
- [ ] Thay mọi số liệu theo bảng số cũ → số mới của Giai đoạn 8; viết lại mục so sánh 2 biến theo 8.2.
- [ ] Checklist TRIPOD (phụ lục), mục Data availability (Zigong theo DUA), cân hai cột trang cuối, ≤ 6 trang.
- [ ] Xác minh lại các tài liệu tham khảo không có trên PubMed (số trang Ishaq 2021, các bài ML, WHO 2011).
- [ ] Mọi con số trong paper truy được về một file trong `reports/` (bảng truy vết `docs/paper/NUMBERS.md`).

### GIAI ĐOẠN 11: CODEX REVIEW LẠI + COMMIT (≈ 0.5 ngày)
- [ ] Codex review code mới/đã sửa trên bản clone sạch **không có dữ liệu Zigong**; gửi thêm các CSV kết quả tổng hợp để đối chiếu với paper.
- [ ] Xử lý mọi phát hiện mức Lớn; ghi các phát hiện còn lại vào `docs/CODEX_REVIEW.md`.
- [ ] Commit theo từng giai đoạn, không có dòng đồng tác giả AI; CI xanh.

### GIAI ĐOẠN 12: THÊM BỘ DỮ LIỆU KIỂM CHỨNG NGOÀI (v1.3; thời gian phụ thuộc thủ tục cấp quyền)

**Ứng viên đã khảo sát:**

| Ưu tiên | Bộ dữ liệu | Quy mô | Kết cục | Biến trùng UCI | Truy cập |
|---|---|---|---|---|---|
| 1 | **MIMIC-III, BN suy tim ICU** (theo Zhou, Li et al., Fudan; bản trích trên Dryad `doi:10.5061/dryad.0p2ngf1zd`) | 1,177 lượt nhập viện, tử vong nội viện 13.5% | Nội viện; **90 ngày** khi tự dựng cohort (có ngày tử vong) | 10/11 (thiếu hút thuốc), có LVEF và CK | PhysioNet credentialed + khóa CITI + DUA MIMIC |
| 2 | **TOPCAT** (HFpEF, EF ≥ 45%) | ≈ 3,400 BN | Tử vong thẩm định, theo dõi nhiều năm | EF, xét nghiệm (đối chiếu danh mục biến) | BioLINCC: đăng ký, miễn IRB, DUA |
| 2 | **SCD-HeFT** (EF ≤ 35%, NYHA II–III) | 2,521 BN | Tử vong | EF, xét nghiệm (đối chiếu) | BioLINCC |
| 3 | HF-ACTION (HFrEF), DIG (bộ nghiên cứu đầy đủ) | ≈ 2,300 / ≈ 6,800 BN | Tử vong / nhập viện | Đối chiếu | BioLINCC |
| 3 | MIMIC-IV, eICU-CRD | Hàng chục nghìn | Nội viện | LVEF gần như không có | PhysioNet credentialed; chỉ cho mô hình không LVEF |

**12.0. Xin quyền truy cập (nhóm thực hiện, làm ngay vì mất nhiều tuần)**
- [ ] MIMIC-III: mỗi thành viên dùng dữ liệu hoàn thành khóa CITI "Data or Specimens Only Research", nâng tài khoản PhysioNet lên credentialed (nếu chưa), ký DUA của MIMIC-III.
- [ ] BioLINCC: đăng ký tài khoản; xin thư miễn IRB (hoặc phê duyệt IRB) của trường; nộp đơn cho TOPCAT và SCD-HeFT. Kiểm tra lại điều kiện vì trang BioLINCC đang thông báo "đang được xem xét để điều chỉnh".
- [ ] Không dùng file Dryad của MIMIC-III cho tới khi người dùng có quyền MIMIC-III; chỉ dùng để đối chiếu số liệu tổng hợp với cohort tự dựng.

**12.1. Dựng cohort MIMIC-III (cần quyền MIMIC-III)**
- [ ] `scripts/build_mimic3_hf_cohort.py`: BN người lớn có mã ICD-9 suy tim (428.x và các mã liên quan, ghi rõ danh sách), **lần nhập ICU đầu tiên mỗi bệnh nhân** (`subject_id` → gom nhóm theo bệnh nhân được, khác Zigong).
- [ ] LVEF lấy từ báo cáo siêu âm tim (`NOTEEVENTS`, category `Echo`) bằng biểu thức chính quy; ghi tỷ lệ trích được và kiểm tra thủ công trên một mẫu nhỏ (chỉ trên máy có quyền MIMIC).
- [ ] Xét nghiệm 24 giờ đầu: creatinine (mg/dL), natri, tiểu cầu (K/µL = 10⁹/L), CK (U/L); bệnh đi kèm từ ICD-9: thiếu máu (280–285), tiểu đường (250.x), tăng huyết áp (401–405); tuổi (> 89 bị dịch trong MIMIC → gán nhóm ≥ 90), giới.
- [ ] Kết cục: tử vong nội viện (`hospital_expire_flag`) và **tử vong 90 ngày** từ ngày nhập viện (`dod`).
- [ ] Đối chiếu tổng hợp với Zhou/Li et al. (≈ 1,177 lượt, 13.5% tử vong nội viện); giải thích chênh lệch.
- [ ] Dữ liệu đặt tại `data/raw/mimic3/` (đã bị `data/raw/*` trong `.gitignore` chặn); không commit, không đưa cho dịch vụ AI.

**12.2. Đồng bộ và phân tích**
- [ ] `harmonize_mimic3()` trong `src/harmonization.py` + bộ sinh dữ liệu giả lập cùng cấu trúc; test như `tests/test_harmonization.py`.
- [ ] Kiểm chứng ngoài UCI → MIMIC-III (90 ngày; all9, no_ef, minimal_two) với CI bootstrap, hiệu chuẩn và hiệu chỉnh trên phần giữ lại (dùng lại code Giai đoạn 9.1).
- [ ] Phân tích gộp 3 cohort và **leave-one-cohort-out**: huấn luyện trên 2 cohort, kiểm chứng trên cohort còn lại; so sánh lại 7 kịch bản xử lý LVEF (MIMIC-III có LVEF đầy đủ hơn Zigong).
- [ ] Khi có TOPCAT / SCD-HeFT: kiểm chứng theo nhóm EF (bảo tồn vs giảm); cân nhắc mở lại Giai đoạn 7 (Cox) với thời gian đến tử vong.

**12.3. Paper**
- [ ] Thêm cohort kiểm chứng ngoài thứ hai vào Methods/Results; trích dẫn MIMIC-III (Johnson et al., Sci Data 2016) và PhysioNet theo yêu cầu của DUA; câu cảm ơn BioLINCC theo mẫu bắt buộc nếu dùng dữ liệu thử nghiệm.
- [ ] Nêu khác biệt bối cảnh: MIMIC-III là BN ICU (Mỹ, 2001–2012), UCI là BN suy tim nặng ngoại trú, Zigong là BN nhập viện (Trung Quốc).

**Nghiệm thu Giai đoạn 12:** script dựng cohort chạy lại được từ dữ liệu gốc; test trên dữ liệu giả lập; CI chỉ dùng dữ liệu giả lập; mọi kết quả công bố là số liệu tổng hợp; không có file dữ liệu nào trong `git ls-files`.

### QUYẾT ĐỊNH CẦN NHÓM CHỐT
1. ~~Có đưa `docs/paper/` lên repo công khai không~~ → **Đã chốt (2026-10-14): không.** `docs/paper/` nằm trong `.gitignore`, chia sẻ nội bộ.
2. Hội nghị mục tiêu và giới hạn số trang → có làm 9.4, 9.5 không.
3. Danh sách tác giả, đơn vị.
4. (v1.3) Thành viên nào xin quyền MIMIC-III và BioLINCC; có chờ dữ liệu BioLINCC trước khi nộp bài không.

---

## 3. CHECKLIST NGHIỆM THU
- [ ] Toàn bộ test PASS, số test ≥ 188 (v1.2); `ruff check src tests scripts` sạch; CI GitHub xanh.
- [ ] `python -m src.train --models logistic_regression random_forest extra_trees mlp` chạy thành công, ghi `reports/model_comparison.csv`.
- [ ] `python -m src.train --feature-set compare` ghi `reports/minimal_vs_full_feature_comparison.csv`.
- [ ] `tests/test_data_leakage.py` vẫn PASS; có test cho mọi bước fit mới (RobustScaler, tập đặc trưng).
- [ ] Không có file Zigong nào trong `git ls-files`.
- [ ] `reports/figures/` có calibration, PR, SHAP (Giai đoạn 5).
- [ ] (v1.2) Mọi phát hiện Lớn của Codex đã sửa, có test; bảng số cũ → số mới đã lập.
- [ ] (v1.2) Mọi con số trong paper truy được về `reports/` (`docs/paper/NUMBERS.md`).

---

## 4. CƠ SỞ CỦA CÁC QUYẾT ĐỊNH
- Zhang Z, et al. *Electronic healthcare records and external outcome data for hospitalized patients with heart failure.* Sci Data 2021;8:46. [DOI](https://doi.org/10.1038/s41597-021-00835-9) — 2,008 BN, 12/2016–06/2019; tử vong khi xuất viện 1% (14/2008); `visit.times` 1 lần 93%, 2 lần 6%, 3–5 lần 1%.
- PhysioNet *heart-failure-zigong* v1.3: restricted access, PhysioNet Restricted Health Data License 1.5.0.
- Sun Z, et al. *Comparing Machine Learning Models and Statistical Models for Predicting Heart Failure Events.* Front Cardiovasc Med 2022;9:812276. [DOI](https://doi.org/10.3389/fcvm.2022.812276) — C-index tử vong: ML 0.777 vs thống kê 0.733; ML không vượt trội nhất quán.
- Số liệu nhãn UCI theo mốc thời gian: tính trực tiếp từ `data/raw/heart_failure_clinical_records_dataset.csv`.
- Van Calster B, et al. *Calibration: the Achilles heel of predictive analytics.* BMC Med 2019;17:230. [DOI](https://doi.org/10.1186/s12916-019-1466-7) — cập nhật mô hình bằng hiệu chỉnh intercept / slope trước khi dùng ở nơi mới.
- Nadeau C, Bengio Y. *Inference for the generalization error.* Mach Learn 2003;52:239–281 — kiểm định t hiệu chỉnh cho CV lặp.
- `docs/CODEX_REVIEW.md` — Codex review 2026-10-14.
- Johnson AEW, et al. *MIMIC-III, a freely accessible critical care database.* Sci Data 2016;3:160035. [DOI](https://doi.org/10.1038/sdata.2016.35).
- Dryad: *Prediction model of in-hospital mortality in ICU patients with heart failure (MIMIC-III).* [doi:10.5061/dryad.0p2ngf1zd](https://datadryad.org/dataset/doi:10.5061/dryad.0p2ngf1zd).
- NHLBI BioLINCC: [TOPCAT](https://biolincc.nhlbi.nih.gov/studies/topcat), [SCD-HeFT](https://biolincc.nhlbi.nih.gov/studies/scd_heft), [HF-ACTION](https://biolincc.nhlbi.nih.gov/studies/hf_action), [DIG](https://biolincc.nhlbi.nih.gov/studies/dig).
