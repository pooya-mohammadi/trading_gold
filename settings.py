# =========================
# 0) Paths & Settings
# =========================
import os
import json
import warnings

warnings.filterwarnings("ignore",
                        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                        category=UserWarning)

CSV_PATH = r"./DataSet_XAUUSD_SP_MTF_TopH1_ReLabeled.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")
OUT_DIR = r"output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ONNX = os.path.join(OUT_DIR, "XAUUSD_SP_MTF_TopH1.onnx")
FEATURE_JSON = os.path.join(OUT_DIR, "XAUUSD_SP_MTF_TopH1.json")
WFV_REPORT_CSV = os.path.join(OUT_DIR, "WalkForward_XAUUSD_SP_MTF_TopH1.csv")
IMP_TOP20_CSV = os.path.join(OUT_DIR, "FeatureImportance_XAUUSD_SP_MTF_TopH1_Top20.csv")
WFV_PLOT_PNG = os.path.join(OUT_DIR, "WFV_Metrics_XAUUSD_SP_MTF_TopH1.png")

ENCODING = "utf-16"  # CSV و خروجی‌های متنی

# ----- Threshold tuning settings (no leakage) -----
THRESH_METHOD = "f1"  # "f1" یا "utility"
VAL_FRAC = 0.10  # 10% انتهایی train همان فولد برای validation

# فقط در حالت "utility" استفاده می‌شوند:
GAIN_PER_TP = 1.0  # سود انتظاری هر TP
COST_PER_FP = 1.0  # ضرر انتظاری هر FP
FEE_PER_TRADE = 0.0  # کارمزد/اسلیپیج هر معامله

# ----- Optional time window filter -----
USE_TIME_WINDOW = True
TIME_START = "2005-09-01"  # مثلاً: "2010-01-01"
TIME_END = "2025-09-01"  # مثلاً: "2018-12-31"
# در صورت USE_TIME_WINDOW=True فقط ردیف‌هایی که OpenTime در بازه [TIME_START, TIME_END) هستند استفاده می‌شوند.

# ----- Reverse, month-based WFV settings -----
# مثال: FOLD_MONTHS=6, VAL_MONTHS_PER_FOLD=1
# یعنی هر فولد 6 ماه؛ 5 ماه اول Train، 1 ماه آخر Test/Verification
FOLD_MONTHS = 36
VAL_MONTHS_PER_FOLD = 2

# =========================
# انتخاب مدل — xgb | lgbm | lr | mlp
# =========================
MODEL_FAMILY = "tabpfn"

# ----- Grid over LGBM hyperparameters -----
USE_GRID = True  # اگر True باشد، چند تنظیم LGBM را پشت سر هم تست می‌کند
LGBM_GRID = [
    {
        "name": "G_safe_120_20",
        "params": {
            "n_estimators": 120,
            "num_leaves": 20,
            "reg_lambda": 2.5,
            "min_data_in_leaf": 180,
        },
    },
    {
        "name": "G_mid_160_22",
        "params": {
            "n_estimators": 160,
            "num_leaves": 22,
            "reg_lambda": 2.0,
            "min_data_in_leaf": 150,
        },
    },
    {
        "name": "G_strong_200_24",
        "params": {
            "n_estimators": 200,
            "num_leaves": 24,
            "reg_lambda": 1.8,
            "min_data_in_leaf": 130,
        },
    },
    {
        "name": "G_xstrong_220_25",
        "params": {
            "n_estimators": 220,
            "num_leaves": 25,
            "reg_lambda": 1.8,
            "min_data_in_leaf": 120,
        },
    },
]

# این‌ها فقط وقتی USE_GRID=True و MODEL_FAMILY=lgbm باشد استفاده می‌شوند.

# =========================
# هدف تصمیم‌گیری
# =========================
TARGET_MODE = "composite"  # "single" | "composite"
TARGET_CLASS = "loss_fast"  # در حالت single
TARGET_GROUP = "profit"  # در حالت composite: "profit" یا "loss"

# =========================
# 1) Classes & RAW features
# =========================
CLASS_ORDER = ["profit_fast", "profit_slow", "loss_fast", "loss_slow"]
CLS2ID = {c: i for i, c in enumerate(CLASS_ORDER)}

# 🔹 دیتاست جدید – فقط همین ستون‌ها به عنوان ورودی مدل استفاده می‌شوند
# "Market_Regime_Stop","Market_Regime_Support"
RAW_FEATURES = [
    "TimeFrame",
    "L12", "L11", "L10", "L9", "L8", "L7", "L6", "L5", "L4", "L3", "L2", "L1",
    "PT12", "PT11", "PT10", "PT9", "PT8", "PT7", "PT6", "PT5", "PT4", "PT3", "PT2", "PT1",
    "PL4", "PL3", "PL2", "PL1",
    "ZT1", "Zig9", "Zig8", "Zig7", "Zig6", "Zig5", "Zig4", "Zig3", "Zig2", "Zig1",
    "ZD9", "ZD8", "ZD7", "ZD6", "ZD5", "ZD4", "ZD3", "ZD2", "ZD1", "Open", "High", "Low", "Close", "OpenPoss", "Sl",
    "Tp",
    "HourOfDay", "DayOfWeek", "Spread", "TickVolume", "Regime"
]
N_RAW = len(RAW_FEATURES)

# ذخیره ترتیب فیچرها برای متاتریدر / ONNX
with open(FEATURE_JSON, "w", encoding=ENCODING) as f:
    json.dump(RAW_FEATURES, f, ensure_ascii=False)

# ❌ دیگر فیچر مشتق داخلی نداریم
DERIVED_NAMES = []
N_DER = 0

CURRENT_LGBM_PARAMS = None  # پارامتر فعال برای LGBM در هر کانفیگ
config_name = "BASE"
# مسیر فایل‌ها برای این کانفیگ
base_wfv_name = os.path.splitext(os.path.basename(WFV_REPORT_CSV))[0]
base_onnx_name = os.path.splitext(os.path.basename(OUT_ONNX))[0]
base_top20_name = os.path.splitext(os.path.basename(IMP_TOP20_CSV))[0]
base_plot_name = os.path.splitext(os.path.basename(WFV_PLOT_PNG))[0]


if USE_GRID and config_name not in (None, "BASE"):
    suffix = f"_{config_name}"
    wfv_csv_path = os.path.join(OUT_DIR, f"{base_wfv_name}{suffix}.csv")
    onnx_path = os.path.join(OUT_DIR, f"{base_onnx_name}{suffix}.onnx")
    top20_path = os.path.join(OUT_DIR, f"{base_top20_name}{suffix}.csv")
    plot_path = os.path.join(OUT_DIR, f"{base_plot_name}{suffix}.png")
else:
    wfv_csv_path = WFV_REPORT_CSV
    onnx_path = OUT_ONNX
    top20_path = IMP_TOP20_CSV
    plot_path = WFV_PLOT_PNG