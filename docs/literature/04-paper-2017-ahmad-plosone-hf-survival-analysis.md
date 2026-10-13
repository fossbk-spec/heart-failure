# Paper Summary: Ahmad et al. (PLoS ONE 2017)

**Tiêu đề**: Survival analysis of heart failure patients: A case study
**Tác giả**: Tanvir Ahmad, Assia Munir, Sajjad Haider Bhatti, Muhammad Aftab, Muhammad Ali Raza (Department of Statistics, Government College University, Faisalabad, Pakistan)
**Tạp chí**: *PLoS ONE* 12(7): e0181001 (2017) — DOI [10.1371/journal.pone.0181001](https://doi.org/10.1371/journal.pone.0181001) — PMID 28727739

> Nguồn trích xuất: abstract PubMed. Số liệu chi tiết (trang/bảng) cần đối chiếu PDF trong `data/SSoT_pdf_references/`.

---

## 1. Mục tiêu & Tập bệnh nhân
- Phân tích sống còn của bệnh nhân suy tim nhập viện tại Institute of Cardiology & Allied Hospital, Faisalabad, giai đoạn 04–12/2015.
- Tiêu chí: ≥ 40 tuổi, rối loạn chức năng tâm thu thất trái, NYHA III–IV. **Đây là nguồn gốc của bộ dữ liệu 299 bệnh nhân dùng trong dự án.**

## 2. Phương pháp
- **Cox regression** với các biến: tuổi, phân suất tống máu, creatinine huyết thanh, natri huyết thanh, thiếu máu, tiểu cầu, CPK, huyết áp, giới, đái tháo đường, hút thuốc.
- Kaplan–Meier cho mô hình sống còn chung; Martingale residuals để đánh giá dạng hàm; kiểm định bằng calibration slope và khả năng phân biệt qua **bootstrapping**; xây dựng **nomogram**.

## 3. Kết quả cốt lõi
- Cường độ tử vong cao ở những ngày đầu, sau đó tăng dần đến cuối nghiên cứu.
- Yếu tố nguy cơ có ý nghĩa: **tuổi, rối loạn chức năng thận (creatinine), huyết áp, phân suất tống máu, thiếu máu**.

## 4. Liên hệ với dự án
- Ở đây `time` là **biến kết cục thời gian** của mô hình sống còn — hợp lệ. Khi chuyển sang bài toán phân loại nhị phân, `time` trở thành rò rỉ và phải loại khỏi X.
