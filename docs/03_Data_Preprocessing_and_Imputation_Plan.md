# Kế Hoạch Tiền Xử Lý Dữ Liệu & Xử Lý Giá Trị Khuyết
## Dự Đoán Tử Vong Do Suy Tim Bằng Dữ Liệu Lớn & Học Máy

---

## 1. Tổng Quan Đặc Tính Hai Bộ Dữ Liệu

| Đặc Tính | Bộ Dữ Liệu 1: UCI / Kaggle | Bộ Dữ Liệu 2: PhysioNet Zigong v1.3 |
|---|---|---|
| **Nguồn gốc** | Faisalabad, Pakistan (Ahmad et al. 2017) | Bệnh viện Số 4 Tự Cống, Tứ Xuyên, Trung Quốc (Zhang et al. 2021) |
| **Số lượng bệnh nhân** | 299 bệnh nhân | 2,008 bệnh nhân nhập viện |
| **Số lượng biến số** | 13 thuộc tính | 168 thuộc tính |
| **Giá trị khuyết** | $0\%$ (Dữ liệu hoàn chỉnh) | Rất cao ở một số biến cận lâm sàng (LVEF: $68.4\%$, CPK: $12.0\%$) |
| **Biến mục tiêu** | `DEATH_EVENT` (0: Sống, 1: Tử vong) — $32.1\%$ tử vong | `DestinationDischarge` (Home, HealthcareFacility, Unknown, Died) |

---

## 2. Đồng Bộ Hóa & Ghép Nối Dữ Liệu (Harmonization Matrix)

Để tiến hành phân tích trên dữ liệu hợp nhất hoặc kiểm chứng chéo (Cross-dataset validation), các biến số cốt lõi được ánh xạ như sau:

| Biến Cốt Lõi | Dạng Dữ Liệu UCI | Dạng Dữ Liệu PhysioNet | Quy Chuẩn Đồng Bộ Hóa |
|---|---|---|---|
| **Tuổi (`age`)** | Số nguyên (năm) | Phân loại theo thập kỷ (Continuous/Decades) | Quy đổi về số liên tục hoặc mã hóa Ordinal Decades. |
| **Phân suất tống máu (`ejection_fraction`)** | Số nguyên ($\%$) | Số liên tục ($\%$) | Đồng bộ về biến liên tục `ejection_fraction` ($\%$). |
| **Serum Creatinine (`serum_creatinine`)** | Số thực ($mg/dL$) | Số thực ($mg/dL$) | Đồng bộ đơn vị $mg/dL$. |
| **Serum Sodium (`serum_sodium`)** | Số nguyên ($mEq/L$) | Số thực ($mEq/L$) | Đồng bộ về biến liên tục $mEq/L$. |
| **Tiểu cầu (`platelets`)** | Số thực ($kiloplatelets/\mu L$) | Số thực ($10^9/L$) | Chuẩn hóa thang đo về cùng đơn vị. |
| **CPK (`creatinine_phosphokinase`)** | Số nguyên ($mcg/L$) | Số thực ($U/L$) | Chuẩn hóa đơn vị đo enzym. |
| **Tăng huyết áp (`high_blood_pressure`)** | Nhị phân ($0/1$) | Nhị phân ($0/1$) | Ánh xạ $0/1$. |
| **Tiểu đường (`diabetes`)** | Nhị phân ($0/1$) | Nhị phân ($0/1$) | Ánh xạ $0/1$. |
| **Biến Mục tiêu (`target`)** | `DEATH_EVENT` ($0/1$) | `DestinationDischarge` | Chuyển đổi thành nhị phân: $1$ nếu Died/Tử vong trong đợt điều trị/theo dõi, $0$ nếu Ra viện/Sống sót. |

---

## 3. Chiến Lược Xử Lý Giá Trị Khuyết (Missing Value Imputation Strategy)

Dựa trên tỷ lệ khuyết thực tế thu được từ EDA:

```
Tỷ lệ khuyết trên dữ liệu gộp (Merged Cohort):
- age, sex, diabetes: 0.0%
- high_blood_pressure: 0.1%
- serum_sodium: 0.5%
- serum_creatinine: 1.1%
- platelets: 1.3%
- cpk: 12.0%
- ejection_fraction: 68.4% (chủ yếu ở PhysioNet)
- smoking: 100.0% (PhysioNet không thu thập smoking)
```

### Mức 1: Giá trị khuyết rất thấp ($< 2\%$)
* **Các biến**: `serum_creatinine` ($1.1\%$), `serum_sodium` ($0.5\%$), `platelets` ($1.3\%$), `high_blood_pressure` ($0.1\%$).
* **Phương pháp**: 
  * Biến liên tục: Sử dụng **Median Imputation** (Trung vị) tính trên tập Train để tránh ảnh hưởng bởi giá trị ngoại lệ.
  * Biến nhị phân: Sử dụng **Mode Imputation** (Yếu vị) trên tập Train.

### Mức 2: Giá trị khuyết trung bình ($5\% - 30\%$)
* **Các biến**: `cpk` ($12.0\%$).
* **Phương pháp**: Sử dụng **MICE (Multivariate Imputation by Chained Equations)** với mô hình Predictive Mean Matching (PMM) hoặc **KNN Imputer** ($k=5$).
* **Quy tắc**: Mô hình MICE/KNNImputer chỉ được `fit` trên tập Train và `transform` trên tập Test.

### Mức 3: Giá trị khuyết nghiêm trọng ($> 50\%$)
* **Các biến**: `ejection_fraction` ($68.4\%$ khuyết ở PhysioNet) và `smoking` ($100\%$ khuyết khi gộp).
* **Phương pháp xử lý**:
  1. **Loại bỏ thuộc tính `smoking`** khỏi bộ dữ liệu hợp nhất vì PhysioNet hoàn toàn thiếu thông tin.
  2. **Đối với `ejection_fraction` ($68.4\%$ missing)**: Gán giá trị cho $68\%$ dữ liệu sẽ tạo ra dữ liệu nhân tạo bị méo lệch nghiêm trọng (fabricating data bias).
  3. **Chiến lược Phân nhánh (Sub-cohort Analysis)**:
     - **Cohort A (Full-Feature Analysis)**: Chỉ phân tích trên tập bệnh nhân có đầy đủ LVEF (299 bệnh nhân UCI + 635 bệnh nhân PhysioNet có LVEF = 934 bệnh nhân).
     - **Cohort B (Broad Imputation / Non-LVEF Model)**: Xây dựng mô hình tiên lượng bổ sung không phụ thuộc vào LVEF cho tập dữ liệu lớn.

---

## 4. Chuẩn Hóa & Biến Đổi Đặc Trưng (Rescaling & Outlier Treatment)

1. **Phát hiện Ngoại Lệ (Outliers)**: Các chỉ số sinh hóa như `creatinine_phosphokinase` và `serum_creatinine` có phân bố lệch phải rất mạnh (right-skewed).
   - Áp dụng biến đổi Logarit ($log(x + 1)$) hoặc **Box-Cox Transformation** cho `cpk` và `serum_creatinine`.
2. **Chuẩn Hóa Thang Đo (Scaling)**:
   - Sử dụng **RobustScaler** (dựa trên Median và IQR) thay vì StandardScaler để hạn chế ảnh hưởng của các giá trị cực trị sinh học.

---

## 5. Phòng Chống Rò Rỉ Dữ Liệu (Data Leakage Prevention) & Cân Bằng Lớp (SMOTE)

Để đảm bảo tính hợp lệ y khoa và tránh thổi phồng kết quả:

### Quy Trình Xử Lý Theo Thứ Tự Bắt Buộc (Strict Pipeline Order):
```
[Dữ liệu Thô] 
     │
     ▼
[1. Tách Tập Dữ Liệu: Train Set (80%) / Test Set (20%)]  <--- TÁCH TRƯỚC TIÊN
     │
     ├──────────────────────────────────────────┐
     ▼                                          ▼
[Tập TRAIN]                                [Tập TEST]
     │                                          │
[2. Học tham số Imputer & Scaler] ─────────────>│ [Áp dụng Imputer & Scaler từ Train]
     │                                          │
[3. Áp dụng SMOTE (chỉ trên Train)]             │ (KHÔNG dùng SMOTE trên Test)
     │                                          │
[4. Huấn luyện Mô hình ML]                      │
     │                                          │
     └───────────────────┬──────────────────────┘
                         ▼
             [5. Đánh giá trên Tập TEST]
```

### Xử Lý Mất Cân Bằng Lớp bằng SMOTE:
- **Nguyên tắc**: SMOTE chỉ được sinh mẫu nhân tạo **trên tập Huấn luyện (Train set)** sau khi đã tách tập Kiểm thử (Test set).
- **Tỷ lệ**: Đưa tỷ lệ lớp thiểu số (tử vong) về mức $50:50$ hoặc $40:60$ trên tập Train để tránh sinh ra quá nhiều dữ liệu nhiễu.

---

## 6. Khung Đánh Giá Mô Hình (Evaluation Metrics)

Do bài toán mang tính chất y khoa với dữ liệu mất cân bằng:
* **Chỉ số Chính (Primary Metrics)**:
  - **PR-AUC (Area Under Precision-Recall Curve)**: Đo lường chính xác hiệu năng trên lớp thiểu số (tử vong).
  - **MCC (Matthews Correlation Coefficient)**: Đánh giá cân bằng cả True Positive và True Negative.
  - **Brier Score / Calibration Curve**: Đánh giá độ hiệu chuẩn xác suất tử vong do mô hình dự báo.
* **Chỉ số Phụ (Secondary Metrics)**:
  - ROC-AUC, Sensitivity (Recall), Specificity, F1-Score.
