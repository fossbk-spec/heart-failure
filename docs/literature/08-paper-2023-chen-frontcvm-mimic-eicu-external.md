# Paper Summary: Chen et al. (Front Cardiovasc Med 2023)

**Tiêu đề**: Machine learning-based in-hospital mortality risk prediction tool for intensive care unit patients with heart failure
**Tác giả**: Zijun Chen, Tingming Li, Sheng Guo, Deli Zeng, Kai Wang (Chongqing Medical University)
**Tạp chí**: *Frontiers in Cardiovascular Medicine* 10:1119699 (2023) — DOI [10.3389/fcvm.2023.1119699](https://doi.org/10.3389/fcvm.2023.1119699) — PMID 37077747

> Nguồn trích xuất: abstract PubMed.

---

## 1. Mục tiêu & Tập bệnh nhân
- Dự đoán tử vong nội viện mọi nguyên nhân của bệnh nhân suy tim sung huyết tại ICU.
- Train: **MIMIC-IV, 11,156 bệnh nhân** (tử vong 13.3%); test **kiểm chứng ngoài**: **eICU-CRD, 9,837 bệnh nhân** (tử vong 13.4%).

## 2. Phương pháp
- LASSO chọn 17 đặc trưng; XGBoost so với Logistic Regression và thang điểm GWTG-HF; đánh giá bằng ROC-AUC, **Brier score**, decision curve; SHAP.

## 3. Kết quả cốt lõi
- Kiểm chứng ngoài: **XGBoost AUC 0.771** (95% CI 0.757–0.784), **Brier 0.100**, vượt LR và GWTG-HF.
- SHAP: APS III, tuổi, SOFA là các yếu tố dự báo mạnh nhất.
- Đã triển khai thành công cụ tính trực tuyến.

## 4. Liên hệ với dự án
- Mẫu mực về **kiểm chứng ngoài giữa hai cơ sở dữ liệu độc lập** — điều bộ dữ liệu 299 bệnh nhân không cho phép; dự án bù lại bằng split theo bệnh nhân + grouped CV và báo cáo Brier score.
