"""
automate_dio.py
===============
Pipeline otomatis preprocessing dataset Telco Customer Churn.
Output: dataset_preprocessed.csv

Author : [Nama Anda]
Date   : 2026-06-01
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    log.info("=" * 60)
    log.info("TAHAP 1 — DATA LOADING")
    log.info("=" * 60)
    log.info(f"Membaca file: {filepath}")

    df = pd.read_csv(filepath)

    log.info(f"Dataset berhasil dimuat.")
    log.info(f"  Shape         : {df.shape}")
    log.info(f"  Jumlah kolom  : {df.shape[1]}")
    log.info(f"  Jumlah baris  : {df.shape[0]}")
    log.info(f"  Kolom         : {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# 2. MISSING VALUE ANALYSIS & HANDLING
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mendeteksi dan menangani missing values."""
    log.info("=" * 60)
    log.info("TAHAP 2 — MISSING VALUE ANALYSIS")
    log.info("=" * 60)

    # Konversi TotalCharges ke numerik (spasi kosong → NaN)
    before = df["TotalCharges"].dtype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    log.info(f"Konversi TotalCharges: {before} → {df['TotalCharges'].dtype}")

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    if missing_cols.empty:
        log.info("Tidak ada missing values setelah konversi.")
    else:
        log.info(f"Missing values terdeteksi pada {len(missing_cols)} kolom:")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            log.info(f"  {col}: {count} nilai hilang ({pct:.2f}%)")

        # Imputasi numerik dengan median
        for col in missing_cols.index:
            if df[col].dtype in [np.float64, np.int64]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.info(f"  → Imputasi '{col}' dengan median = {median_val:.4f}")
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                log.info(f"  → Imputasi '{col}' dengan modus = {mode_val}")

    log.info(f"Total missing values tersisa: {df.isnull().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
# 3. DUPLICATE ANALYSIS
# ---------------------------------------------------------------------------
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Mendeteksi dan menghapus duplikat."""
    log.info("=" * 60)
    log.info("TAHAP 3 — DUPLICATE ANALYSIS")
    log.info("=" * 60)

    dup_count = df.duplicated().sum()
    log.info(f"Jumlah baris duplikat ditemukan: {dup_count}")

    if dup_count > 0:
        df = df.drop_duplicates()
        log.info(f"  → {dup_count} baris duplikat dihapus.")
        log.info(f"  → Shape setelah hapus duplikat: {df.shape}")
    else:
        log.info("Tidak ada duplikat. Dataset tetap bersih.")

    return df


# ---------------------------------------------------------------------------
# 4. OUTLIER ANALYSIS & HANDLING (IQR Capping)
# ---------------------------------------------------------------------------
def _iqr_bounds(series: pd.Series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Mendeteksi outlier dengan IQR dan melakukan capping (Winsorization)."""
    log.info("=" * 60)
    log.info("TAHAP 4 — OUTLIER ANALYSIS (IQR Capping)")
    log.info("=" * 60)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    for col in num_cols:
        lower, upper = _iqr_bounds(df[col])
        n_outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        log.info(
            f"  {col}: {n_outliers} outlier "
            f"| batas bawah={lower:.2f}, batas atas={upper:.2f}"
        )
        df[col] = df[col].clip(lower=lower, upper=upper)
        remaining = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        log.info(f"    → Setelah capping: {remaining} outlier tersisa")

    return df


# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur-fitur baru dari data yang sudah ada."""
    log.info("=" * 60)
    log.info("TAHAP 5 — FEATURE ENGINEERING")
    log.info("=" * 60)

    # Fitur 1: Rata-rata charge per bulan
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    log.info("  [+] AvgMonthlyCharge = TotalCharges / (tenure + 1)")

    # Fitur 2: Jumlah layanan aktif
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    no_service_vals = {"no", "no internet service", "no phone service"}

    def _count_services(row):
        return sum(
            1 for col in service_cols
            if str(row[col]).lower() not in no_service_vals
        )

    df["NumServices"] = df.apply(_count_services, axis=1)
    log.info("  [+] NumServices = jumlah layanan yang aktif digunakan")

    # Fitur 3: HasFamily
    df["HasFamily"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)
    log.info("  [+] HasFamily = 1 jika memiliki Partner atau Dependents")

    # Fitur 4: TenureGroup
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        right=True,
    )
    log.info("  [+] TenureGroup = segmen tenure dalam kategori bulan")
    log.info(f"  Distribusi TenureGroup:\n{df['TenureGroup'].value_counts().to_string()}")

    return df


# ---------------------------------------------------------------------------
# 6. ENCODING
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah fitur kategorikal menjadi representasi numerik."""
    log.info("=" * 60)
    log.info("TAHAP 6 — ENCODING")
    log.info("=" * 60)

    df = df.copy()

    # Hapus kolom ID
    df.drop(columns=["customerID"], inplace=True)
    log.info("  Kolom 'customerID' dihapus (bukan fitur prediktif).")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    log.info("  Target 'Churn': Yes→1, No→0")

    # Label Encoding — kolom biner
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        log.info(f"  LabelEncoder → '{col}'")

    # One-Hot Encoding — kolom multi-kategori
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod", "TenureGroup",
    ]
    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    after_cols = df.shape[1]
    log.info(f"  OneHotEncoding → {len(ohe_cols)} kolom menghasilkan {after_cols - before_cols + len(ohe_cols)} kolom baru.")

    # Pastikan bool → int
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        log.info(f"  Konversi bool→int pada {len(bool_cols)} kolom.")

    log.info(f"  Shape setelah encoding: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 7. SCALING
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi fitur numerik menggunakan StandardScaler."""
    log.info("=" * 60)
    log.info("TAHAP 7 — SCALING (StandardScaler)")
    log.info("=" * 60)

    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyCharge", "NumServices"]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    for col in scale_cols:
        log.info(
            f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}"
        )

    log.info(f"  Scaling selesai pada {len(scale_cols)} kolom numerik.")
    return df


# ---------------------------------------------------------------------------
# 8. SIMPAN OUTPUT
# ---------------------------------------------------------------------------
def save_output(df: pd.DataFrame, output_path: str) -> None:
    """Menyimpan dataset hasil preprocessing ke file CSV."""
    log.info("=" * 60)
    log.info("TAHAP 8 — MENYIMPAN DATASET")
    log.info("=" * 60)

    df.to_csv(output_path, index=False)
    log.info(f"  Dataset disimpan ke : {output_path}")
    log.info(f"  Shape final         : {df.shape}")
    log.info(f"  Jumlah fitur        : {df.shape[1] - 1}  (+ 1 target 'Churn')")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """Menjalankan seluruh pipeline preprocessing secara berurutan."""
    log.info("*" * 60)
    log.info("  MEMULAI PIPELINE PREPROCESSING — automate_dio.py")
    log.info("*" * 60)

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = handle_outliers(df)
    df = feature_engineering(df)
    df = encode_features(df)
    df = scale_features(df)
    save_output(df, output_path)

    log.info("*" * 60)
    log.info("  PIPELINE SELESAI — telco_churn_preprocessed.csv siap digunakan.")
    log.info("*" * 60)
    return df


if __name__ == "__main__":
    INPUT_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_FILE = "telco_churn_preprocessed.csv"
    run_pipeline(INPUT_FILE, OUTPUT_FILE)
