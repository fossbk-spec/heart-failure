# KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT (ACTION PLAN) CHO CLAUDE CODE / CODEX
## Dự án: Heart Failure Mortality Prediction using Big Data & Machine Learning
**Đường dẫn thư mục dự án:** `D:\Antigravity\heart-failure`
**Repo:** https://github.com/fossbk-spec/heart-failure
**Phiên bản:** v1.1 (2026-10-13) — sửa đổi từ v1.0 sau khi review đối chiếu với mã nguồn và tài liệu gốc

---

## 0. THAY ĐỔI SO VỚI v1.0

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
- **Kết quả (paired grouped CV 5 fold trên train+val, Recall ≥ 0.80):** mô hình 2 biến không kém mô hình 11 biến; ROC-AUC RF 0.78 → 0.83, Extra Trees 0.75 → 0.83, GB 0.74 → 0.78 (CI hiệu số chỉ loại trừ 0 ở GB). Ủng hộ Chicco & Jurman (2020) về hướng, chưa đủ mạnh về thống kê với 239 BN.

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
  3. **Cohort A**: chỉ BN có LVEF (thiếu không ngẫu nhiên → báo cáo thiên lệch chọn mẫu). **Cohort B**: bỏ LVEF, thêm biến chỉ báo thiếu; không nội suy 68% LVEF.
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

### GIAI ĐOẠN 5: BIỂU ĐỒ & TRÍCH DẪN
- [ ] `scripts/generate_clinical_figures.py` (300 DPI, `reports/figures/`): calibration curve (LR / RF / ETC, có và không SMOTE) + calibration slope/intercept + Brier; đường PR; SHAP summary & bar.
- [ ] `data/SSoT_pdf_references/references.bib`: gộp 2 bộ tài liệu, có DOI. Xác minh với PDF:
  - Sun 2022: 202 + 78 là số **nghiên cứu**; con số "97% High Risk of Bias" không có trong abstract.
  - Zhang 2021: abstract ghi 166 thuộc tính, PhysioNet ghi 168.
  - Kapelios 2023: xác minh tên tạp chí.

### GIAI ĐOẠN 6: ROBUSTSCALER (ưu tiên thấp)
- [ ] `make_preprocessor(scaler="standard" | "robust")`, mặc định `standard`. Lợi ích nhỏ vì đã có log1p.

### GIAI ĐOẠN 7: PHÂN TÍCH SỐNG CÒN (tùy chọn, độ nhạy)
- [ ] Cox PH (lifelines hoặc scikit-survival), C-index, AUC phụ thuộc thời gian tại 90 ngày; `time` là thời gian kết cục.

---

## 3. CHECKLIST NGHIỆM THU
- [ ] Toàn bộ test PASS, số test ≥ 105; `ruff check` sạch; CI GitHub xanh.
- [ ] `python -m src.train --models logistic_regression random_forest extra_trees mlp` chạy thành công, ghi `reports/model_comparison.csv`.
- [ ] `python -m src.train --feature-set compare` ghi `reports/minimal_vs_full_feature_comparison.csv`.
- [ ] `tests/test_data_leakage.py` vẫn PASS; có test cho mọi bước fit mới (RobustScaler, tập đặc trưng).
- [ ] Không có file Zigong nào trong `git ls-files`.
- [ ] `reports/figures/` có calibration, PR, SHAP (Giai đoạn 5).

---

## 4. CƠ SỞ CỦA CÁC QUYẾT ĐỊNH
- Zhang Z, et al. *Electronic healthcare records and external outcome data for hospitalized patients with heart failure.* Sci Data 2021;8:46. [DOI](https://doi.org/10.1038/s41597-021-00835-9) — 2,008 BN, 12/2016–06/2019; tử vong khi xuất viện 1% (14/2008); `visit.times` 1 lần 93%, 2 lần 6%, 3–5 lần 1%.
- PhysioNet *heart-failure-zigong* v1.3: restricted access, PhysioNet Restricted Health Data License 1.5.0.
- Sun Z, et al. *Comparing Machine Learning Models and Statistical Models for Predicting Heart Failure Events.* Front Cardiovasc Med 2022;9:812276. [DOI](https://doi.org/10.3389/fcvm.2022.812276) — C-index tử vong: ML 0.777 vs thống kê 0.733; ML không vượt trội nhất quán.
- Số liệu nhãn UCI theo mốc thời gian: tính trực tiếp từ `data/raw/heart_failure_clinical_records_dataset.csv`.
