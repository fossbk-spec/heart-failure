# Paper Summary: Wilstrup & Cave (BMC Med Inform Decis Mak 2022)

**Tiêu đề**: Combining symbolic regression with the Cox proportional hazards model improves prediction of heart failure deaths
**Tác giả**: Casper Wilstrup, Chris Cave (Abzu, Copenhagen)
**Tạp chí**: *BMC Medical Informatics and Decision Making* 22:196 (2022) — DOI [10.1186/s12911-022-01943-1](https://doi.org/10.1186/s12911-022-01943-1) — PMID 35879758

> Nguồn trích xuất: abstract PubMed. Chỉ số C-index cụ thể cần đối chiếu PDF trong SSoT.

---

## 1. Mục tiêu & Dữ liệu
- Cùng bộ 299 bệnh nhân suy tim Pakistan. Mục tiêu: cải thiện mô hình Cox bằng các phép biến đổi phi tuyến tìm được tự động.

## 2. Phương pháp
- **QLattice** (symbolic regression) tìm biến đổi toán học của các hiệp biến, sau đó đưa vào mô hình **Cox proportional hazards**.

## 3. Kết quả cốt lõi
- Yếu tố nguy cơ tốt nhất: **hàm mũ của tuổi**, **nghịch đảo của ejection fraction**, **nghịch đảo của serum creatinine**.
- Cox với hiệp biến đã biến đổi có hiệu năng dự báo tốt hơn Cox trên hiệp biến gốc.

## 4. Liên hệ với dự án
- Ủng hộ việc biến đổi phi tuyến các xét nghiệm lệch (dự án dùng `log1p` cho creatinine, CPK, tiểu cầu) và cho thấy mô hình đơn giản, diễn giải được vẫn cạnh tranh.
