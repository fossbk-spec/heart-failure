# Heart Failure Clinical Records Dataset Specification

**Nguồn**: UCI Machine Learning Repository #519 (bản phân phối của Chicco & Jurman 2020); dữ liệu gốc: Ahmad et al. (2017), Institute of Cardiology & Allied Hospital, Faisalabad, Pakistan (04–12/2015)
**Mirror**: Kaggle `andrewmvd/heart-failure-clinical-data`
**Quy mô**: 299 bệnh nhân (1 bản ghi/bệnh nhân), 13 thuộc tính
**Giấy phép**: CC BY 4.0
**Tiêu chí chọn bệnh nhân** (Ahmad 2017): ≥ 40 tuổi, rối loạn chức năng tâm thu thất trái, NYHA III–IV.

---

## 1. Cấu trúc các thuộc tính (Attribute Information)

| STT | Tên thuộc tính | Kiểu dữ liệu | Mô tả chi tiết | Dùng làm đặc trưng? |
|-----|----------------|--------------|----------------|----------------------|
| — | `patient_id` | Identifier | KHÔNG có trong CSV; `data_loader` thêm 1..299 theo thứ tự file | Không (định danh) |
| 1 | `age` | Continuous | Tuổi (năm), 40–95 | Có |
| 2 | `anaemia` | Binary | Giảm hồng cầu/hemoglobin (0/1) | Có |
| 3 | `creatinine_phosphokinase` | Continuous | Men CPK trong máu (mcg/L), lệch phải mạnh (23–7,861) | Có (log1p) |
| 4 | `diabetes` | Binary | Đái tháo đường (0/1) | Có |
| 5 | `ejection_fraction` | Continuous (nguyên) | Phân suất tống máu thất trái LVEF (%), 14–80 | Có |
| 6 | `high_blood_pressure` | Binary | Tăng huyết áp (0/1) | Có |
| 7 | `platelets` | Continuous | Tiểu cầu (kiloplatelets/mL) | Có (log1p) |
| 8 | `serum_creatinine` | Continuous | Creatinine huyết thanh (mg/dL), 0.5–9.4 | Có (log1p) |
| 9 | `serum_sodium` | Continuous (nguyên) | Natri huyết thanh (mEq/L), 113–148 | Có |
| 10 | `sex` | Binary | 1 = nam, 0 = nữ | Có |
| 11 | `smoking` | Binary | Hút thuốc (0/1) | Có |
| 12 | `time` | Integer | Thời gian theo dõi (ngày), 4–285 | **KHÔNG — rò rỉ thời gian** |
| 13 | `DEATH_EVENT` | Binary (Target) | 1: tử vong trong thời gian theo dõi, 0: còn sống/kiểm duyệt | Nhãn |

---

## 2. Thách thức chất lượng dữ liệu (đo trực tiếp bằng `scripts/run_eda.py`)
- **Mất cân bằng lớp vừa phải**: 203 sống (67.9%) / 96 tử vong (32.1%), tỷ lệ 2.1 : 1.
- **Không có giá trị NaN**, nhưng có dấu hiệu **nội suy ẩn trước khi công bố**:
  `platelets = 263358.03` (đúng bằng trung bình cột) lặp lại 25 lần; `creatinine_phosphokinase = 582` lặp lại 47 lần.
- **Rò rỉ thời gian (`time`)**: trung vị theo dõi 172 ngày (sống) vs 44.5 ngày (tử vong); AUC đơn biến 0.84 — cao hơn mọi đặc trưng nền tảng (cao nhất: `serum_creatinine` 0.73). Tử vong chấm dứt theo dõi nên `time` chứa thông tin của nhãn.
- **Cỡ mẫu nhỏ**: tập test 20% chỉ có 60 bệnh nhân / 19 ca tử vong → kết quả test dao động lớn; ưu tiên grouped CV.
- **Thiên lệch nhân khẩu học**: nam 64.9% / nữ 35.1%; một trung tâm, một quốc gia, một năm → khả năng khái quát hóa hạn chế.
