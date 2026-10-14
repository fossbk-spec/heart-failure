# TIẾP TỤC DỰ ÁN TRÊN MÁY KHÁC

Toàn bộ mã nguồn, tài liệu, notebook và kế hoạch đã nằm trên GitHub. Tiến độ xem tại [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md). Tài liệu này liệt kê những gì **không** có trong git và cách dựng lại chúng trên máy mới.

## 1. Những gì không nằm trong git

| Thành phần | Lý do | Cách khôi phục |
|---|---|---|
| `.venv/` | Môi trường ảo, phụ thuộc vào từng máy | `make setup` (mục 2) |
| `data/raw/zigong/dat.csv` | Dữ liệu hạn chế, bị cấm chia sẻ lại theo DUA | Tự tải bằng tài khoản PhysioNet (mục 3) |
| `reports/*.csv`, `*.txt`, `figures/*.png`, `eda/`, `recall80/`, `external_validation/` | Kết quả sinh ra từ mã nguồn | Chạy lại (mục 4) |
| `docs/paper/` (bản thảo bài báo, script build) | Giữ riêng tư cho đến khi nộp bài | Lấy từ kênh nội bộ của nhóm |
| `data/SSoT_pdf_references/*.pdf` | Bản quyền thuộc nhà xuất bản | Tải lại theo `data/SSoT_pdf_references/README.md` |

## 2. Cài đặt môi trường

```bash
git clone https://github.com/fossbk-spec/heart-failure.git
cd heart-failure
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows
# .venv/bin/python -m pip install -r requirements.txt       # Linux / macOS
```

Các lỗi đã gặp trên máy cũ (Windows) và cách xử lý:
- **pip báo lỗi SSL** khi máy có Avast chặn TLS: thêm
  `--cert "C:/ProgramData/Avast Software/Avast/wscert.pem"` vào lệnh pip.
- **Không dùng môi trường base của Anaconda:** gói `imblearn` trong đó bị lỗi import. Luôn dùng `.venv`.
- **Chạy notebook bằng nbconvert:** cài kernel của venv rồi trỏ Jupyter tới đó.
  ```bash
  .venv/Scripts/python -m ipykernel install --sys-prefix --name python3
  JUPYTER_PATH=$(pwd)/.venv/share/jupyter .venv/Scripts/jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
  ```
- **Log có tiếng Việt hoặc ký hiệu Unicode:** đặt `PYTHONIOENCODING=utf-8`.

Kiểm tra môi trường: `make verify` và `make test` (172 test phải qua).

## 3. Dữ liệu Zigong

```bash
python scripts/download_zigong.py --user <tên-tài-khoản-physionet>
python scripts/download_zigong.py --verify-only   # 2.008 dòng, 42 ca tử vong 90 ngày, sha256 trùng khớp
```

- Chỉ người đã được PhysioNet xác thực và đã ký DUA mới được tải và sử dụng dữ liệu này.
- Không commit dữ liệu, không đưa lên CI, Colab hay Kaggle công khai.
- Nếu chép file trực tiếp giữa các máy, chỉ dùng phương tiện được mã hoá hoặc có kiểm soát quyền truy cập. Sau khi chép, chạy lại `--verify-only` để kiểm tra.

## 4. Tái tạo kết quả

```bash
make eda                   # reports/eda/, reports/figures/eda_overview.png
make train                 # reports/model_comparison.csv, cv_*.csv, train_log.txt
make train-recall80        # reports/recall80/
make compare-features      # reports/minimal_vs_full_feature_comparison.csv
make inspect-zigong        # thống kê tổng hợp của Zigong
make external-validation   # reports/external_validation/ (khoảng vài phút với 1.000 lần bootstrap)
```

- Trên Windows không có `make`: chạy trực tiếp các lệnh trong `Makefile`, thay `.venv/bin/python` bằng `.venv/Scripts/python`.
- Mọi kết quả đều dùng `random_state` cố định, nên số liệu tái tạo được phải khớp với README và `PROGRESS_REPORT.md`.

## 5. Quy ước làm việc

- **Không ghi nhận AI trên GitHub:**
  - commit và PR không có dòng `Co-Authored-By: Claude`, "Generated with Claude Code", hay bất kỳ ghi nhận AI nào;
  - tài liệu trong `docs/` được phép nhắc đến cả Claude và Codex.
- **Giữ Zero Data Leakage:**
  - biến `time` không bao giờ được dùng làm đặc trưng;
  - mọi bước tiền xử lý chỉ fit trên tập train;
  - SMOTE luôn nằm trong Pipeline.
- **Dữ liệu Zigong:** chỉ công bố số liệu tổng hợp, không bao giờ in hay lưu dữ liệu ở mức từng bệnh nhân vào repo.

## 6. Điểm tiếp tục

Giai đoạn 0–4 đã xong. Việc tiếp theo theo [`ACTION_PLAN_FOR_CLAUDE.md`](ACTION_PLAN_FOR_CLAUDE.md):
1. **Giai đoạn 5:**
   - biểu đồ hiệu chuẩn: UCI có và không có SMOTE, Zigong trước và sau khi hiệu chỉnh;
   - đường PR và SHAP ở 300 DPI;
   - file `references.bib` kèm DOI.
2. **Giai đoạn 6:** tuỳ chọn RobustScaler.
3. **Giai đoạn 7 (tuỳ chọn):** phân tích sống còn bằng mô hình Cox.
4. **Việc của nhóm:** tải lại từ điển biến của Zigong để xác nhận đơn vị CK.
