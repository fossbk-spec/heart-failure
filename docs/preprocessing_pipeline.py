import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, matthews_corrcoef, make_scorer

# Try importing imblearn, fallback if not present
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = False


def create_dummy_heart_failure_data(n_samples=500, random_state=42):
    """
    Tạo bộ dữ liệu giả lập mô phỏng đặc tính kết hợp UCI + PhysioNet Zigong
    để chạy thử nghiệm độc lập.
    """
    np.random.seed(random_state)
    
    age = np.random.normal(loc=60, scale=12, size=n_samples)
    ejection_fraction = np.random.normal(loc=38, scale=11, size=n_samples)
    # Tạo giá trị khuyết cho LVEF (30% missing)
    mask_ef = np.random.rand(n_samples) < 0.30
    ejection_fraction[mask_ef] = np.nan
    
    serum_creatinine = np.random.exponential(scale=1.2, size=n_samples) + 0.5
    serum_sodium = np.random.normal(loc=136, scale=4, size=n_samples)
    cpk = np.random.exponential(scale=250, size=n_samples) + 50
    mask_cpk = np.random.rand(n_samples) < 0.10
    cpk[mask_cpk] = np.nan
    
    high_blood_pressure = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    diabetes = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    
    # Target nhãn tử vong (Mất cân bằng lớp ~ 30% tử vong)
    logit = (0.03 * age - 0.05 * (ejection_fraction - 40) + 0.8 * serum_creatinine 
             + 0.5 * high_blood_pressure - 3.0)
    # Điền giá trị tạm thời cho logit nếu EF missing để tính prob
    ef_clean = np.nan_to_num(ejection_fraction, nan=38)
    logit_clean = (0.03 * age - 0.05 * (ef_clean - 40) + 0.8 * serum_creatinine 
                   + 0.5 * high_blood_pressure - 3.0)
    prob = 1 / (1 + np.exp(-logit_clean))
    target = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'ejection_fraction': ejection_fraction,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'cpk': cpk,
        'high_blood_pressure': high_blood_pressure,
        'diabetes': diabetes,
        'DEATH_EVENT': target
    })
    return df


def build_preprocessing_pipeline(num_features_knn, num_features_simple, cat_features, apply_smote=True):
    """
    Xây dựng Pipeline xử lý dữ liệu chuẩn tránh Rò rỉ thông tin (Data Leakage).
    
    Các bước trong Pipeline:
    1. Imputation (Gán giá trị khuyết)
    2. Scaling (Chuẩn hóa)
    3. SMOTE (Chỉ cân bằng lớp trong lúc fit tập Train)
    4. Classifier (Mô hình phân loại)
    """
    # 1. Transformers cho các nhóm đặc trưng
    num_knn_transformer = Pipeline_step = ImbPipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    
    num_simple_transformer = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_transformer = ImbPipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # 2. Ghép nối ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_knn', num_knn_transformer, num_features_knn),
            ('num_simple', num_simple_transformer, num_features_simple),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='drop'
    )
    
    # 3. Tạo Full Pipeline kết hợp SMOTE và Classifier
    pipeline_steps = [('preprocessor', preprocessor)]
    
    if apply_smote and HAS_IMBLEARN:
        pipeline_steps.append(('smote', SMOTE(random_state=42)))
    
    pipeline_steps.append(('classifier', ExtraTreesClassifier(n_estimators=100, random_state=42)))
    
    full_pipeline = ImbPipeline(pipeline_steps)
    return full_pipeline


def run_pipeline_demonstration():
    print("=" * 70)
    print("DEMO PIPELINE TIỀN XỬ LÝ & PHÂN CHIA DỮ LIỆU TRÁNH RÒ RỈ THÔNG TIN")
    print("=" * 70)
    
    # Step 1: Khởi tạo dữ liệu
    df = create_dummy_heart_failure_data(n_samples=600, random_state=42)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    print(f"[-] Kích thước dữ liệu gốc: {X.shape}")
    print(f"[-] Tỷ lệ phân bố lớp nhãn: {np.bincount(y)} (Tử vong: {y.mean():.1%})\n")
    
    # Step 2: ĐĨNH NGHĨA NHÓM ĐẶC TRƯNG
    num_features_knn = ['ejection_fraction', 'cpk']
    num_features_simple = ['age', 'serum_creatinine', 'serum_sodium']
    cat_features = ['high_blood_pressure', 'diabetes']
    
    # Step 3: TÁCH TẬP TRAIN / TEST ĐẦU TIÊN (PREVENT LEAKAGE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[✓] Tách dữ liệu Train/Test trước tiên thành công:")
    print(f"    - Tập Train: {X_train.shape[0]} mẫu")
    print(f"    - Tập Test:  {X_test.shape[0]} mẫu (Được giữ nguyên làm Holm/Out-of-sample)")
    
    # Step 4: THIẾT LẬP CROSS-VALIDATION TRÊN TẬP TRAIN
    pipeline = build_preprocessing_pipeline(
        num_features_knn, num_features_simple, cat_features, apply_smote=HAS_IMBLEARN
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'f1': 'f1',
        'mcc': make_scorer(matthews_corrcoef)
    }
    
    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False
    )
    
    print("\n[✓] Kết quả 5-Fold Cross-Validation trên tập Train (An toàn, không rò rỉ):")
    print(f"    - Mean ROC-AUC : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"    - Mean PR-AUC  : {cv_results['test_pr_auc'].mean():.4f} ± {cv_results['test_pr_auc'].std():.4f}")
    print(f"    - Mean F1-Score: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"    - Mean MCC     : {cv_results['test_mcc'].mean():.4f} ± {cv_results['test_mcc'].std():.4f}")
    
    # Step 5: FIT TRÊN TOÀN BỘ TẬP TRAIN & ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    test_roc = roc_auc_score(y_test, y_prob)
    test_pr = average_precision_score(y_test, y_prob)
    test_mcc = matthews_corrcoef(y_test, y_pred)
    
    print("\n[✓] Đánh giá trên tập Hold-out Test Độc lập (External Holdout):")
    print(f"    - Test ROC-AUC : {test_roc:.4f}")
    print(f"    - Test PR-AUC  : {test_pr:.4f}")
    print(f"    - Test MCC     : {test_mcc:.4f}")
    print("\nBáo cáo chi tiết Classification Report trên tập Test:")
    print(classification_report(y_test, y_pred, target_names=['Sống', 'Tử vong']))
    
    print("=" * 70)


if __name__ == '__main__':
    run_pipeline_demonstration()
