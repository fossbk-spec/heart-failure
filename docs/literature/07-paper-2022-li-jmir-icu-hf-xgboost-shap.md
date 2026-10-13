# Paper Summary: Li et al. (J Med Internet Res 2022)

**Tiêu đề**: Predicting Mortality in Intensive Care Unit Patients With Heart Failure Using an Interpretable Machine Learning Model: Retrospective Cohort Study
**Tác giả**: Jili Li, Siru Liu, Yundi Hu, Lingfeng Zhu, Yujia Mao, Jialin Liu
**Tạp chí**: *Journal of Medical Internet Research* 24(8): e38082 (2022) — DOI [10.2196/38082](https://doi.org/10.2196/38082) — PMID 35943767

> Nguồn trích xuất: abstract PubMed.

---

## 1. Mục tiêu & Tập bệnh nhân
- Dự đoán tử vong nội viện của bệnh nhân suy tim tại ICU; **2,798 bệnh nhân** từ eICU Collaborative Research Database; tỷ lệ tử vong **9.97%**.

## 2. Phương pháp
- Dữ liệu 24 giờ đầu nhập ICU; **chia ngẫu nhiên 70/30**; XGBoost so với 3 mô hình ML khác; SHAP để giải thích; decision curve analysis.

## 3. Kết quả cốt lõi
- **XGBoost AUC 0.824** (95% CI 0.7766–0.8708); SVM kém nhất (AUC 0.701).
- Net benefit của XGBoost vượt các mô hình khác ở ngưỡng xác suất 10%–28%.
- SHAP: **blood urea nitrogen trung bình** là yếu tố dự báo quan trọng nhất.

## 4. Hạn chế & khoảng trống
- Chia ngẫu nhiên; abstract không nêu việc gom nhóm theo bệnh nhân khi một bệnh nhân có nhiều lần nhập ICU (cần kiểm tra trong PDF).
- Không kiểm chứng ngoài.
