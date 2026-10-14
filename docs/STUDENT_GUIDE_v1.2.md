# HƯỚNG DẪN THỰC HIỆN VÀ KIỂM TRA CHO SINH VIÊN — KẾ HOẠCH v1.2

**Dự án:** Dự đoán tử vong ở bệnh nhân suy tim bằng Machine Learning
**Repo:** https://github.com/fossbk-spec/heart-failure
**Kế hoạch gốc:** [`ACTION_PLAN_FOR_CLAUDE.md`](ACTION_PLAN_FOR_CLAUDE.md) v1.3, Giai đoạn 8–12
**Tài liệu nên đọc trước:**
- [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md): những gì đã làm;
- [`CODEX_REVIEW.md`](CODEX_REVIEW.md): các lỗi cần sửa;
- [`HANDOVER_NEW_MACHINE.md`](HANDOVER_NEW_MACHINE.md): cài đặt môi trường;
- bản thảo bài báo `docs/paper/`: **không có trên repo công khai** (bị `.gitignore` chặn cho đến khi nộp bài); nhóm chia sẻ qua kênh nội bộ, sinh viên đặt vào `docs/paper/` trên máy mình.

Tài liệu này là **gợi ý**: nó chỉ ra cần làm gì, gợi ý hướng làm và cách tự kiểm tra, nhưng không đưa sẵn lời giải. Mỗi nhiệm vụ gồm năm phần:
1. **Mục tiêu**;
2. **Gợi ý**;
3. **Tự kiểm tra**: lệnh và kết quả mong đợi;
4. **Tiêu chí nghiệm thu**: người hướng dẫn dùng để chấm;
5. **Câu hỏi hiểu bài**: sinh viên phải trả lời được khi bảo vệ.

---

## 0. Trước khi bắt đầu

### 0.1. Quy tắc bắt buộc
1. **Dữ liệu Zigong:**
   - Chỉ sinh viên có tài khoản PhysioNet đã xác thực và đã ký DUA mới được tải và chạy trên dữ liệu thật (`python scripts/download_zigong.py --user <tên>`).
   - Sinh viên khác làm việc trên dữ liệu giả lập (`--synthetic`).
   - Không commit, không gửi qua chat hay email, không tải lên Colab/Kaggle công khai, không dán dữ liệu vào ChatGPT, Claude, Codex hay bất kỳ dịch vụ AI nào.
   - Chỉ được chia sẻ **số liệu tổng hợp** (AUC, tỷ lệ, số ca).
2. **Zero Data Leakage:** mọi bước có `fit` (imputer, scaler, log1p, SMOTE, hiệu chỉnh mô hình) chỉ được học trên dữ liệu huấn luyện. Biến `time` không bao giờ được dùng làm đặc trưng.
3. **Không làm gãy test:** trước mỗi commit phải chạy
   ```bash
   .venv/Scripts/python -m pytest -q
   .venv/Scripts/python -m ruff check src tests scripts
   ```
4. **Git:**
   - Mỗi nhiệm vụ một nhánh (`fix/8.1-calibration-solver`, `feat/9.1-recalibration`, …), merge qua Pull Request và cần một bạn khác review.
   - Không ghi dòng đồng tác giả AI trong commit hay PR.
5. **Dùng trợ lý AI** (Claude, Codex) để hỗ trợ code thì được, nhưng sinh viên phải hiểu và giải thích được mọi dòng mình nộp (xem các câu hỏi hiểu bài). Không đưa dữ liệu Zigong cho AI.

### 0.2. Chuẩn bị môi trường (≈ 1 giờ)
- [ ] Làm theo [`HANDOVER_NEW_MACHINE.md`](HANDOVER_NEW_MACHINE.md).
- [ ] `make test` (hoặc lệnh pytest ở trên) cho kết quả **188 passed**.
- [ ] Sinh viên có quyền truy cập Zigong: `python scripts/download_zigong.py --verify-only` báo đạt cả 3 điều kiện kiểm tra.
- [ ] Chạy lại và lưu kết quả **trước khi sửa** (dùng cho bảng số cũ → số mới):
  ```bash
  make train-recall80 compare-features external-validation pooled-missing
  ```
  Sau đó sao chép thư mục `reports/` thành `reports_before_v1.2/` (thư mục này không commit).

### 0.3. Gợi ý phân công (4 sinh viên, ≈ 2 tuần)

| Sinh viên | Nhiệm vụ | Cần Zigong? |
|---|---|---|
| SV1 | 8.1, 8.4, 9.1 (hiệu chuẩn và hiệu chỉnh mô hình) | 9.1 cần |
| SV2 | 8.2, 8.5 (thống kê so sánh, CLI, test) | Không |
| SV3 | 8.3, 8.6, 9.3 (stacking, odds ratio, độ nhạy) | 9.3 cần |
| SV4 | 9.2, Giai đoạn 10 (phân tích gộp, paper, truy vết số liệu) | 9.2 cần |
| SV có quyền MIMIC-III | Giai đoạn 12.1–12.2 (v1.3) | Cần MIMIC-III |
| Cả nhóm | Giai đoạn 11 (Codex review lại, nghiệm thu chung) | — |

**Thứ tự phụ thuộc:**
- Giai đoạn 8 phải xong trước 9.1 và 9.2, vì 9.1 dùng hàm hiệu chuẩn đã sửa ở 8.1.
- Giai đoạn 10 làm sau cùng, khi đã có số liệu cuối.

---

## 1. GIAI ĐOẠN 8 — Sửa lỗi theo Codex review

### 8.1. Hàm hiệu chuẩn hội tụ ổn định (SV1, ưu tiên cao)
**Mục tiêu:** `calibration_intercept_slope` và `recalibrate_intercept` trong `src/external_validation.py` luôn trả về kết quả hữu hạn, hoặc báo rõ là không hội tụ.

**Gợi ý:**
- Intercept khi coi `logit(p)` là offset là nghiệm của phương trình một ẩn: Σ(yᵢ − σ(a + logit pᵢ)) = 0. Vế trái đơn điệu giảm theo `a`, nên dùng `scipy.optimize.brentq` trên một khoảng như [−20, 20].
- Slope: dùng `scipy.optimize.minimize` (BFGS) trên log-likelihood âm, hoặc Newton có line search. Kiểm tra `result.success`.
- Khi chỉ có một lớp, hoặc dữ liệu tách hoàn toàn: trả NaN kèm cờ `converged=False`, không ném lỗi khó hiểu.

**Tự kiểm tra (viết test trước khi sửa, test phải fail với code cũ):**
- `p = 0.99` cho mọi dòng, `y` có tỷ lệ 2%. Khi đó CITL phải bằng logit(0,02) − logit(0,99) ≈ −8,49 (sai số 0,01).
- Dữ liệu hiệu chuẩn hoàn hảo (như test hiện có): intercept ≈ 0, slope ≈ 1.
- `recalibrate_intercept(y, p).mean()` ≈ `y.mean()`, kể cả với `p` cực đoan.

**Nghiệm thu:**
- [ ] Có ≥ 3 test mới, gồm kịch bản dự đoán cực đoan mà Codex mô tả.
- [ ] Kết quả trên Zigong thật không đổi quá 0,01 so với `reports_before_v1.2/`. Nếu đổi, phải giải thích được.

**Câu hỏi hiểu bài:**
- Vì sao Newton không có line search có thể bước quá xa?
- CITL = −3,4 nghĩa là gì về mặt lâm sàng?

### 8.2. So sánh 2 biến với 11 biến một cách chặt chẽ (SV2, ưu tiên cao)
**Mục tiêu:** thay khoảng tin cậy t thông thường trong `compare_feature_sets` (`src/train.py`) bằng một phương pháp có tính đến việc các fold chồng lên nhau.

**Gợi ý:**
- Chạy CV lặp: r = 10 lần, k = 5 fold, tạo các fold bằng `StratifiedGroupKFold` với seed thay đổi theo từng lần lặp. Hai tập đặc trưng phải dùng **cùng một bộ fold**.
- Kiểm định t hiệu chỉnh của Nadeau–Bengio: với J = r·k hiệu số dⱼ, phương sai hiệu chỉnh là σ̂² = (1/J + n_test/n_train) · s², trong đó s² là phương sai mẫu của các dⱼ. Thống kê t = d̄ / σ̂, bậc tự do J − 1.
- Xuất các cột `mean_diff`, `corrected_se`, `corrected_ci_low/high`, `p_value`.

**Tự kiểm tra:**
- Hai tập đặc trưng giống nhau thì mọi hiệu số bằng 0.
- Khoảng tin cậy hiệu chỉnh phải **rộng hơn** khoảng t thường trên cùng dữ liệu.

**Nghiệm thu:**
- [ ] Có test cho công thức (so với kết quả tính tay trên một mảng nhỏ).
- [ ] README, `PROGRESS_REPORT.md` và paper chỉ dùng lời mô tả ("điểm ước lượng tương đương hoặc cao hơn"), không còn chữ "không kém".

**Câu hỏi hiểu bài:**
- Vì sao 5 hiệu số từ 5 fold không độc lập với nhau?
- Muốn kết luận "non-inferiority" thì cần những gì?

### 8.3. Stacking chia fold nội bộ theo bệnh nhân (SV3)
**Gợi ý:**
- `StackingClassifier(cv=...)` nhận được một danh sách các cặp (train_idx, test_idx).
- Cách đơn giản và an toàn: trong `fit`, nếu `patient_id` có giá trị trùng lặp thì báo lỗi rõ ràng.
- Cách đầy đủ hơn: tạo các fold bằng `StratifiedGroupKFold` theo nhóm bệnh nhân rồi truyền vào.

**Tự kiểm tra:** tạo dữ liệu giả lập trong đó một bệnh nhân có 2 dòng. Phải bị báo lỗi, hoặc hai dòng đó luôn nằm cùng một fold.

**Nghiệm thu:**
- [ ] Có test.
- [ ] Kết quả stacking trên UCI không đổi, vì mỗi bệnh nhân UCI chỉ có một dòng.

### 8.4. Log1p theo tên cột và xử lý cohort rỗng (SV1)
**Gợi ý:**
- Dùng lại cách của `make_pooled_preprocessor` trong `src/pooled_missing.py`: dùng `ColumnTransformer` chọn cột theo tên **trước** bước impute.
- Trong `validate_model`, nếu không còn dòng nào áp dụng được thì trả về một hàng chỉ số toàn NaN, với `zigong_n = 0`.

**Tự kiểm tra:**
- Một cột numeric thiếu toàn bộ: log1p vẫn áp dụng đúng vào CPK, creatinine và tiểu cầu.
- Đặt toàn bộ LVEF của Zigong giả lập thành NaN rồi chạy `validate_model(feature_set="minimal_two")`: không lỗi, trả NaN.

**Nghiệm thu:** có 2 test; mọi test cũ vẫn pass.

### 8.5. CLI, test spy và tham số `beta` (SV2)
**Gợi ý:**
- `--models` dùng `choices=["logistic_regression", "random_forest", "extra_trees"]`.
- Test spy: đổi index của cohort Zigong thành `+100000` để hai cohort không trùng index, rồi kiểm tra mọi `X` truyền vào `fit` và vào `threshold_for_min_recall` chỉ có index của UCI.
- Tên metric F-beta lấy theo `f"f{beta:g}"`.

**Tự kiểm tra:**
- `python -m src.train --feature-set compare --beta 1` chạy được.
- Sửa thử để pipeline fit trên `zigong.iloc[:len(uci)]`: test spy mới phải **fail**.

### 8.6. Khoảng tin cậy odds ratio và mô tả kiểm tra rò rỉ (SV3, ưu tiên thấp)
**Gợi ý:**
- Thêm cột hằng số 1 vào ma trận thiết kế, nghịch đảo ma trận thông tin Fisher đầy đủ, rồi lấy khối ứng với các hệ số.
- Sửa README: nói rõ `assert_no_fit_on_holdout` chỉ kiểm tra rằng `transform` không làm thay đổi trạng thái đã fit, chứ không chứng minh pipeline chưa từng được fit trên hold-out.

**Tự kiểm tra:** với dữ liệu không có SMOTE và không có penalty, CI phải khớp với `statsmodels.Logit` trong khoảng ±0,01.

### Nghiệm thu chung Giai đoạn 8
- [ ] Chạy lại 4 lệnh `make` ở mục 0.2, rồi lập `docs/paper/NUMBERS_CHANGES.md` gồm các cột: vị trí (README hay paper, bảng nào), số cũ, số mới, lý do thay đổi.
- [ ] Toàn bộ test pass, ruff sạch, CI xanh.

---

## 2. GIAI ĐOẠN 9 — Bổ sung phân tích

### 9.1. Hiệu chỉnh trên một phần Zigong, đánh giá trên phần còn lại (SV1, cần Zigong, ưu tiên cao)
**Mục tiêu:** trả lời câu hỏi: nếu bệnh viện mới có một ít dữ liệu của chính mình, việc hiệu chỉnh lại mô hình UCI cải thiện được bao nhiêu?

**Gợi ý quy trình (một lần lặp):**
1. Chia Zigong ngẫu nhiên 50/50, phân tầng theo kết cục: nửa đầu để hiệu chỉnh, nửa sau để đánh giá. Zigong không có ngày nhập viện nên không chia theo thời gian được.
2. Mô hình UCI (LR, 8 biến, không LVEF) đã fit sẵn, giữ nguyên.
3. Bốn cách:
   - (a) giữ nguyên mô hình;
   - (b) chỉ hiệu chỉnh intercept trên phần hiệu chỉnh;
   - (c) hiệu chỉnh cả intercept và slope (hồi quy logistic của y theo logit p̂);
   - (d) huấn luyện lại trên UCI cộng phần hiệu chỉnh, có biến cohort.
4. Trên phần đánh giá, tính AUC, CITL, slope và Brier.
5. Lặp lại ≥ 200 lần, báo cáo trung vị và khoảng 2,5–97,5%.

**Lưu ý chống rò rỉ:** các tham số hiệu chỉnh ở (b), (c) và mô hình huấn luyện lại ở (d) **chỉ** được học trên phần hiệu chỉnh.

**Tự kiểm tra trên dữ liệu giả lập:**
- (b) đưa trung vị CITL trên phần đánh giá về gần 0.
- AUC của (a), (b), (c) bằng nhau trong từng lần lặp, vì biến đổi đơn điệu không thay đổi thứ hạng.

**Nghiệm thu:**
- [ ] `src/recalibration.py`, `scripts/run_recalibration.py`, test, lệnh `make recalibration`.
- [ ] CSV tổng hợp nằm trong `reports/recalibration/`, thư mục này được thêm vào `.gitignore`.

**Câu hỏi hiểu bài:**
- Vì sao hiệu chỉnh intercept không làm thay đổi AUC?
- Với 21 ca tử vong ở phần hiệu chỉnh, vì sao (c) có thể kém (b)?

### 9.2. Khoảng tin cậy cho phân tích gộp (SV4, cần Zigong)
**Gợi ý:**
- Thêm tham số `n_repeats=10` cho `run_pooled_scenarios`.
- CI cho AUC trong từng cohort: bootstrap phân tầng trên dự đoán out-of-fold của từng lần lặp (dùng lại `bootstrap_auc_ci`), rồi gộp bằng trung vị các cận.
- Báo cáo thêm hiệu số AUC "bỏ cột LVEF" trừ "điền MICE" trong Zigong, kèm CI. Hai cách phải dùng cùng fold.

**Nghiệm thu:** bảng 7 kịch bản có CI; kết luận ở Giai đoạn 4b của kế hoạch được xác nhận lại hoặc sửa.

### 9.3. Chỉ giữ lượt nhập viện đầu tiên (SV3, cần Zigong)
**Gợi ý:**
- Thêm tham số `first_visit_only` cho `harmonize_zigong`, lọc `visit.times == 1`.
- Chạy lại kiểm chứng ngoài và phân tích gộp với tham số này.

**Nghiệm thu:** bảng so sánh kết quả có và không có bộ lọc; ghi thành một câu trong phần Limitations của paper.

### 9.4, 9.5 (tuỳ chọn, chỉ làm khi nhóm quyết định)
- **9.4:** huấn luyện trên Zigong, kiểm chứng trên UCI, chỉ dùng 8 biến không có LVEF. Phải ghi rõ chỉ có 42 ca tử vong để huấn luyện.
- **9.5:** delta adjustment. Sau khi điền LVEF, cộng thêm δ ∈ {−10, −5, +5, +10} vào các giá trị điền, rồi xem AUC trong Zigong thay đổi thế nào.

---

## 2b. GIAI ĐOẠN 12 — Thêm bộ dữ liệu kiểm chứng ngoài (v1.3)

Làm song song với Giai đoạn 8–10, vì thủ tục cấp quyền mất nhiều tuần.

### 12.0. Xin quyền truy cập (làm ngay, cả nhóm)
- **MIMIC-III:** hoàn thành khóa CITI "Data or Specimens Only Research", nâng tài khoản PhysioNet lên credentialed nếu chưa có, rồi ký DUA của MIMIC-III.
- **BioLINCC (TOPCAT, SCD-HeFT):** đăng ký tài khoản, xin thư miễn IRB của trường, nộp đơn kèm DUA.
- **Tự kiểm tra:** mỗi người đăng nhập được trang MIMIC-III trên PhysioNet và tải được một file nhỏ như `PATIENTS.csv.gz`.

### 12.1. Dựng cohort MIMIC-III (một SV có quyền MIMIC)
**Gợi ý:**
- Dựng cohort từ dữ liệu gốc, không dùng file Dryad để huấn luyện. File Dryad chỉ dùng để đối chiếu số liệu tổng hợp.
- Mỗi bệnh nhân chỉ lấy lần nhập ICU đầu tiên. MIMIC có `subject_id`, nên khác với Zigong, ở đây gom nhóm theo bệnh nhân được.
- LVEF lấy từ báo cáo siêu âm tim bằng biểu thức chính quy (ví dụ "EF 35%", "LVEF 30-35%"). Khoảng giá trị thì lấy điểm giữa. Ghi lại tỷ lệ trích được.
- Nhãn 90 ngày tính từ `dod` và ngày nhập viện. Tuổi trên 89 trong MIMIC đã bị dịch đi, nên gán vào nhóm ≥ 90.

**Tự kiểm tra:**
- Số lượt nhập viện và tỷ lệ tử vong nội viện gần với Zhou/Li et al. (≈ 1.177 lượt, 13,5%). Nếu lệch hơn 10% phải giải thích được vì sao.
- Kiểm tra thủ công 20 báo cáo siêu âm ngẫu nhiên: LVEF trích bằng regex phải khớp với văn bản. Việc này chỉ làm trên máy có quyền MIMIC, và tuyệt đối không dán báo cáo vào công cụ AI.

**Nghiệm thu:**
- [ ] `scripts/build_mimic3_hf_cohort.py` chạy lại được từ dữ liệu gốc.
- [ ] Hàm `harmonize_mimic3()` có test trên dữ liệu giả lập.
- [ ] Bảng mô tả cohort chỉ chứa số liệu tổng hợp.

### 12.2. Phân tích
- Kiểm chứng ngoài UCI → MIMIC-III. Dùng lại `src/external_validation.py` và phần hiệu chỉnh ở 9.1.
- Leave-one-cohort-out trên 3 cohort: huấn luyện trên 2 cohort, kiểm chứng trên cohort còn lại.
- Chạy lại 7 kịch bản xử lý LVEF.

**Câu hỏi hiểu bài:**
- Vì sao kết quả trên bệnh nhân ICU ở Mỹ không thay được kết quả trên Zigong?
- Leave-one-cohort-out trả lời câu hỏi gì mà CV trên dữ liệu gộp không trả lời được?

## 3. GIAI ĐOẠN 10 — Cập nhật kết quả vào bài báo (SV4, cả nhóm review)

**Nguyên tắc:** mọi con số trong paper phải truy được về một file trong `reports/`.

1. **Lập bảng truy vết `docs/paper/NUMBERS.md`:** mỗi dòng gồm con số, vị trí trong paper (ví dụ "Table IV, LR no-LVEF AUC"), file nguồn trong `reports/`, cột và dòng tương ứng.
2. **Cập nhật số liệu:** sửa nội dung trong `docs/paper/build_paper.py` rồi build lại; không sửa tay file `.docx`. Lệnh build ghi trong `docs/paper/README.md`.
3. **Các mục cần cập nhật:**
   - [ ] Mọi số liệu thay đổi sau Giai đoạn 8 (theo `NUMBERS_CHANGES.md`).
   - [ ] Mục so sánh 2 biến: viết theo kiểm định Nadeau–Bengio (8.2).
   - [ ] Mục "Pooled Analysis and Missing LVEF" (đã có bản nháp): thêm CI từ 9.2.
   - [ ] Mục mới về hiệu chỉnh (9.1); sửa Discussion và Conclusion cho phù hợp.
   - [ ] Limitations: thêm kết quả 9.3.
   - [ ] Data availability: UCI theo CC BY 4.0; Zigong là dữ liệu hạn chế, cần tài khoản PhysioNet và DUA; mã nguồn công khai.
   - [ ] Checklist TRIPOD (22 mục) làm phụ lục hoặc tài liệu kèm theo.
   - [ ] Tên tác giả, đơn vị; câu cảm ơn về việc dùng trợ lý AI theo quy định của hội nghị.
4. **Định dạng:**
   - ≤ 6 trang; cân hai cột ở trang cuối;
   - bảng và hình được đánh số và được nhắc đến trong bài;
   - tài liệu tham khảo đánh số theo thứ tự xuất hiện.
5. **Kiểm tra trích dẫn:** đối chiếu DOI và số trang trên trang của nhà xuất bản, đặc biệt [Ishaq 2021], các bài ML (Breiman, Geurts, Friedman, Wolpert, Lundberg) và tài liệu WHO 2011.

**Nghiệm thu:**
- [ ] Người hướng dẫn chọn ngẫu nhiên 10 con số trong paper và truy được cả 10 về `reports/`.
- [ ] Một sinh viên không tham gia viết đọc lại toàn bài, kiểm tra thuật ngữ và tính nhất quán giữa Abstract, Results và Conclusion.

---

## 4. GIAI ĐOẠN 11 — Codex review lại và nghiệm thu chung

1. Tạo bản clone sạch, không có dữ liệu Zigong:
   ```bash
   git clone <repo> ../hf-review
   ```
   Kiểm tra rằng thư mục `../hf-review/data/raw/zigong/` không tồn tại.
2. Cho Codex review ở chế độ read-only. Prompt mẫu nằm trong `docs/CODEX_REVIEW.md`; hãy mở rộng phạm vi sang các file mới (`src/recalibration.py`, …).
3. Gửi cho Codex các CSV tổng hợp trong `reports/` cùng `docs/paper/NUMBERS.md` để đối chiếu số liệu. **Không gửi dữ liệu Zigong.**
4. Sửa mọi phát hiện mức Lớn. Các phát hiện còn lại ghi vào `docs/CODEX_REVIEW.md` (phần "Lần 2"), kèm quyết định sửa hay không và lý do.

---

## 5. Bảng kiểm tra dành cho người hướng dẫn

| # | Kiểm tra | Cách kiểm tra | Đạt |
|---|---|---|---|
| 1 | Test và lint | `pytest -q` (≥ 188 test + test mới), `ruff check src tests scripts`, CI xanh | ☐ |
| 2 | Không lộ dữ liệu | `git log --all --stat \| grep -i zigong` chỉ ra code và tài liệu, không có file dữ liệu | ☐ |
| 3 | Lỗi Codex đã sửa | Mỗi mục 8.1–8.6 có test tái hiện lỗi; đổi lại code cũ thì test fail | ☐ |
| 4 | Không rò rỉ trong 9.1 | Đọc code: tham số hiệu chỉnh chỉ học trên phần hiệu chỉnh; có test spy | ☐ |
| 5 | Số cũ → số mới | `NUMBERS_CHANGES.md` đầy đủ, có lý do cho mỗi thay đổi lớn hơn 0,01 | ☐ |
| 6 | Truy vết số liệu | Chọn ngẫu nhiên 10 số trong paper, tất cả có trong `NUMBERS.md` và đúng với `reports/` | ☐ |
| 7 | Diễn đạt thống kê | Không có "non-inferior" hay "không kém" khi chưa có kiểm định phù hợp; AUC gộp không được dùng làm kết luận chính | ☐ |
| 8 | Hiểu bài | Mỗi sinh viên trả lời được các câu hỏi hiểu bài của nhiệm vụ mình làm | ☐ |
| 9 | Git | Mỗi nhiệm vụ có một PR được review; không có dòng đồng tác giả AI | ☐ |

## 6. Lịch gợi ý (2 tuần)

| Ngày | Việc |
|---|---|
| 1 | Mục 0: môi trường, lưu `reports_before_v1.2/`, phân công |
| 2–4 | Giai đoạn 8 (song song SV1–SV3); SV4 lập `NUMBERS.md` cho bản paper hiện tại |
| 5 | Merge Giai đoạn 8, chạy lại, lập `NUMBERS_CHANGES.md` |
| 6–8 | Giai đoạn 9 (9.1, 9.2, 9.3) |
| 9–11 | Giai đoạn 10: cập nhật paper |
| 12 | Giai đoạn 11: Codex review lại, sửa lỗi |
| 13–14 | Nghiệm thu theo bảng mục 5, chuẩn bị nộp |
