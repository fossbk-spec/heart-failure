# Paper Summary: Tanaka et al. (ESC Heart Failure 2024)

**Tiêu đề**: Development of interpretable machine learning models to predict in-hospital prognosis of acute heart failure patients
**Tác giả**: Munekazu Tanaka, Hirohiko Kohjitani, Erika Yamamoto, … Koh Ono (Kyoto University và các bệnh viện thuộc KCHF registry)
**Tạp chí**: *ESC Heart Failure* 11(5): 2798–2812 (2024) — DOI [10.1002/ehf2.14834](https://doi.org/10.1002/ehf2.14834) — PMID 38751135

> Nguồn trích xuất: abstract PubMed.

---

## 1. Mục tiêu & Tập bệnh nhân
- Dự đoán tử vong nội viện và suy tim nặng lên (WHF) ở bệnh nhân suy tim cấp; Kyoto Congestive Heart Failure registry, **4,056 bệnh nhân** (sau loại trừ: 4,040).
- Tỷ lệ tử vong nội viện **6.3%**; WHF **22.3%**.

## 2. Phương pháp
- Chỉ dùng thông tin ngày nhập viện đầu tiên; chia **4:1** train/test (3,232 / 808); CART, Random Forest, XGBoost; đánh giá sensitivity, specificity, AUC, **Brier score, calibration slope**; SHAP + phân cụm diễn giải.

## 3. Kết quả cốt lõi
- Tử vong nội viện: **XGBoost AUC 0.816** vs RF 0.755 vs CART 0.683.
- WHF: XGBoost AUC 0.766 vs RF 0.713 vs CART 0.688.

## 4. Liên hệ với dự án
- Chỉ dùng dữ liệu tại thời điểm nhập viện — đúng tinh thần loại `time` khỏi đặc trưng; báo cáo calibration bên cạnh discrimination.
