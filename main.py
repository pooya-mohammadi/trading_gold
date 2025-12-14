import warnings

import numpy as np
import pandas as pd

from settings import (CSV_PATH, ENCODING, RAW_FEATURES, USE_TIME_WINDOW, TIME_START, TIME_END, CLASS_ORDER,
                      TARGET_CLASS, TARGET_MODE, TARGET_GROUP, CLS2ID, MODEL_FAMILY, VAL_FRAC, FOLD_MONTHS,
                      VAL_MONTHS_PER_FOLD, THRESH_METHOD, GAIN_PER_TP, COST_PER_FP, FEE_PER_TRADE, USE_GRID, LGBM_GRID,
                      DERIVED_NAMES, N_RAW, N_DER, wfv_csv_path, onnx_path, FEATURE_JSON, plot_path,
                      top20_path)
from utils import make_base_pipeline, estimate_overfit_risk

warnings.filterwarnings("ignore",
                        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                        category=UserWarning)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.base import clone

# ====== Base models ======

try:
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

# ====== ONNX toolchains ======
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OXFloatTensorType

try:
    from onnxmltools import convert_lightgbm

    _HAS_CONVERT_LGBM = True
except Exception:
    _HAS_CONVERT_LGBM = False

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SKLFloatTensorType

import matplotlib.pyplot as plt

first_row = pd.read_csv(CSV_PATH, nrows=1, encoding=ENCODING)
parse_cols = [c for c in ["OpenTime"] if c in first_row.columns]
df = pd.read_csv(CSV_PATH, parse_dates=parse_cols, encoding=ENCODING)

missing_feats = [c for c in RAW_FEATURES if c not in df.columns]
if missing_feats:
    raise ValueError(f"Missing required RAW_FEATURES: {missing_feats}")

# -------------------------------
# 👀 بررسی و فیلتر اولیه مقادیر TimeFrame (مثل کد اول)
# -------------------------------
if "TimeFrame" not in df.columns:
    raise ValueError("TimeFrame column is missing in the CSV!")

print("🔍 Unique TimeFrame values found in dataset (raw):")
print(sorted(df["TimeFrame"].unique()))

# ALLOWED_TF براساس کلیدهای tf_mapping جدید (شامل 20)
ALLOWED_TF = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 16385}

# تبدیل به عدد صحیح (اگر به‌صورت استرینگ ذخیره شده باشد)
tf_all = pd.to_numeric(df["TimeFrame"], errors="coerce").astype("Int64")
invalid_mask = ~tf_all.isin(ALLOWED_TF) & tf_all.notna()

n_before = len(df)
n_drop = int(invalid_mask.sum())

if n_drop > 0:
    invalid_vals = sorted(tf_all[invalid_mask].unique().tolist())
    print(f"⚠️ Dropping {n_drop} rows with unsupported TimeFrame values: {invalid_vals}")

# فقط سطرهای مجاز را نگه می‌داریم
mask_keep = tf_all.isin(ALLOWED_TF)
df = df[mask_keep].copy()
df["TimeFrame"] = tf_all[mask_keep].astype(int)

n_after = len(df)
print(f"✅ TimeFrame filtering: kept {n_after} of {n_before} rows.")

# -------------------------------
# 🕒 فیلتر بازه زمانی (در صورت فعال بودن)
# -------------------------------
if USE_TIME_WINDOW:
    if "OpenTime" not in df.columns:
        raise ValueError("OpenTime column is required when USE_TIME_WINDOW=True")

    t_start = pd.to_datetime(TIME_START) if TIME_START else None
    t_end = pd.to_datetime(TIME_END) if TIME_END else None

    mask_time = pd.Series(True, index=df.index)
    if t_start is not None:
        mask_time &= (df["OpenTime"] >= t_start)
    if t_end is not None:
        mask_time &= (df["OpenTime"] < t_end)

    n_before_tw = len(df)
    df = df[mask_time].copy()
    n_after_tw = len(df)
    print(
        f"✅ Time window filtering: kept {n_after_tw} of {n_before_tw} rows "
        f"between {t_start.date() if t_start is not None else '-∞'} and "
        f"{t_end.date() if t_end is not None else '+∞'}."
    )

label_col = "Label_4C"
if label_col not in df.columns:
    raise ValueError("Label_4C column not found in CSV.")

y_col = df[label_col]
if y_col.dtype == object:
    bad = [u for u in y_col.dropna().unique() if u not in CLASS_ORDER]
    if bad:
        raise ValueError(f"Unexpected label names in Label_4C: {bad}. Expected subset of {CLASS_ORDER}")
    y_all_full = y_col.map(CLS2ID).astype(np.int64).to_numpy()
else:
    uniq = sorted(pd.unique(y_col.dropna()))
    if len(uniq) != 4:
        raise ValueError(f"Label_4C must have 4 unique values; found {len(uniq)}: {uniq}")
    value2id = {v: i for i, v in enumerate(uniq)}
    y_all_full = y_col.map(value2id).astype(np.int64).to_numpy()

if "OpenTime" not in df.columns:
    raise ValueError("OpenTime column is required for time-based walk-forward splits.")

df_sorted = df.sort_values("OpenTime").reset_index(drop=True)
X_all_df = df_sorted[RAW_FEATURES].astype(np.float32)
X_all_np_full = X_all_df.to_numpy(dtype=np.float32, copy=False)  # NumPy اصلی برای تمام fit/predict
y_all_full = y_all_full[df_sorted.index]

y_all_full = y_all_full.astype(np.int64)
if (y_all_full == -1).any():
    y_all_full = np.where(y_all_full == -1, 3, y_all_full)
u = np.unique(y_all_full)
if not set(u).issubset({0, 1, 2, 3}):
    raise ValueError(f"Label_4Cs must be in 0..3 after mapping; found {u}")

# =========================
# هدف دودویی
# =========================
IDX = {"profit_fast": 0, "profit_slow": 1, "loss_fast": 2, "loss_slow": 3}


def make_target_scores(proba: np.ndarray, y_int: np.ndarray):
    if TARGET_MODE.lower() == "single":
        if TARGET_CLASS not in IDX:
            raise ValueError(f"TARTGET_CLASS must be one of {list(IDX.keys())}")
        t_idx = IDX[TARGET_CLASS]
        p = proba[:, t_idx]
        y = (y_int == t_idx).astype(int)
        label_name = TARGET_CLASS
    else:
        grp = TARGET_GROUP.lower()
        if grp not in ("profit", "loss"):
            raise ValueError('TARGET_GROUP must be "profit" or "loss"')
        if grp == "profit":
            p = proba[:, IDX["profit_fast"]] + proba[:, IDX["profit_slow"]]
            y = np.isin(y_int, [IDX["profit_fast"], IDX["profit_slow"]]).astype(int)
            label_name = "profit_composite"
        else:
            p = proba[:, IDX["loss_fast"]] + proba[:, IDX["loss_slow"]]
            y = np.isin(y_int, [IDX["loss_fast"], IDX["loss_slow"]]).astype(int)
            label_name = "loss_composite"
    return p, y, label_name


# =========================
# تابع اجرای کامل برای یک کانفیگ
# =========================
def run_for_config(config_name="BASE", lgbm_params=None):
    # global CURRENT_LGBM_PARAMS
    # CURRENT_LGBM_PARAMS = lgbm_params

    # کپی لوکال از داده‌ها
    X_all_np = X_all_np_full
    y_all = y_all_full
    times = df_sorted["OpenTime"]

    # 3) Walk-Forward (reverse, month-based) + threshold tuning
    base_pipe = make_base_pipeline(MODEL_FAMILY, VAL_FRAC)
    rows = []

    # times همین الان بر اساس OpenTime سورت شده
    times = pd.to_datetime(times)
    data_start = times.min()
    data_end_actual = times.max()

    # انتهای مؤثر دیتاست (بعد از فیلتر Time Window)
    dataset_end = data_end_actual

    # فولدها را از انتهای دیتاست (dataset_end) به سمت گذشته می‌سازیم
    current_end = dataset_end
    fold_idx = 0

    while True:
        fold_idx += 1

        # انتهای فولد = current_end
        # شروع فولد = current_end - FOLD_MONTHS
        # شروع بخش Test در فولد = current_end - VAL_MONTHS_PER_FOLD
        fold_start_raw = current_end - pd.DateOffset(months=FOLD_MONTHS)
        val_start_raw = current_end - pd.DateOffset(months=VAL_MONTHS_PER_FOLD)

        # کلیپ به داخل محدوده واقعی دیتاست
        fold_start = max(fold_start_raw, data_start)
        val_start = max(val_start_raw, fold_start)

        # ماسک Train: از fold_start تا قبل از val_start
        train_mask = (times >= fold_start) & (times < val_start)
        # ماسک Test/Verification: از val_start تا قبل از current_end
        test_mask = (times >= val_start) & (times < current_end)

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        # اگر دیگر هیچ دیتایی در این بازه نبود، حلقه را تمام کن
        if n_train == 0 and n_test == 0:
            break

        # اگر فقط یکی از train/test صفر بود، این فولد را رد کن و برو عقب‌تر
        if n_train == 0 or n_test == 0:
            current_end = fold_start
            if current_end <= data_start:
                break
            continue

        # ---- فولد معتبر: Train روی چند ماه اول، Test روی ماه آخر ----
        train_mask_np = train_mask.to_numpy()
        test_mask_np = test_mask.to_numpy()

        X_train_np = X_all_np[train_mask_np]
        X_test_np = X_all_np[test_mask_np]
        y_train = y_all[train_mask_np]
        y_test = y_all[test_mask_np]

        train_pipe = clone(base_pipe)
        train_pipe.fit(X_train_np, y_train)

        # --- تخمین ریسک اوورفیت برای این فولد ---
        clf_fold = train_pipe.named_steps["clf"]
        overfit_info = estimate_overfit_risk(clf_fold, n_train_samples=n_train, verbose=False)

        # ---- threshold tuning روی تهِ TRAIN همان فولد ----
        proba_tr = train_pipe.predict_proba(X_train_np)  # (N,4)
        p_tr, y_bin_tr, label_name = make_target_scores(proba_tr, y_train)

        n_tr = len(p_tr)
        cut = int(max(1, n_tr * (1.0 - VAL_FRAC)))
        tau = 0.5
        if n_tr >= 50 and cut < n_tr:
            p_va, y_va = p_tr[cut:], y_bin_tr[cut:]
            if np.unique(y_va).size >= 2:
                prec_, rec_, thr_ = precision_recall_curve(y_va, p_va)
                if len(thr_) > 0:
                    if THRESH_METHOD.lower() == "utility":
                        def utility_at(t):
                            y_hat = (p_va >= t).astype(int)
                            TP = ((y_va == 1) & (y_hat == 1)).sum()
                            FP = ((y_va == 0) & (y_hat == 1)).sum()
                            Trades = y_hat.sum()
                            return TP * GAIN_PER_TP - FP * COST_PER_FP - Trades * FEE_PER_TRADE

                        utils = [utility_at(t) for t in thr_]
                        tau = float(thr_[int(np.argmax(utils))])
                    else:
                        f1s = (2 * prec_[:-1] * rec_[:-1]) / (prec_[:-1] + rec_[:-1] + 1e-12)
                        best_idx = int(np.argmax(f1s))
                        tau = float(thr_[best_idx])

        # ---- evaluate on TEST/Verification (ماه آخر فولد) ----
        proba_te = train_pipe.predict_proba(X_test_np)  # (N_test, 4)
        y_pred_4c = proba_te.argmax(axis=1)

        acc_4c = accuracy_score(y_test, y_pred_4c)
        f1m_4c = f1_score(y_test, y_pred_4c, average="macro")

        p_te, y_bin_true = make_target_scores(proba_te, y_test)[:2]
        y_bin_pred = (p_te >= tau).astype(int)

        acc_bin = accuracy_score(y_bin_true, y_bin_pred)
        f1_bin = f1_score(y_bin_true, y_bin_pred)
        prec_bin = precision_score(y_bin_true, y_bin_pred, zero_division=0)
        rec_bin = recall_score(y_bin_true, y_bin_pred, zero_division=0)

        rows.append({
            "fold_index": fold_idx,
            "train_start": fold_start,
            "train_end": val_start,
            "test_start": val_start,
            "test_end": current_end,
            "n_train": n_train,
            "n_test": n_test,
            "acc_4class": acc_4c,
            "f1_macro_4c": f1m_4c,
            "acc_binary": acc_bin,
            "precision_binary": prec_bin,
            "recall_binary": rec_bin,
            "f1_binary": f1_bin,
            "threshold_used": float(tau),
            "th_method": THRESH_METHOD,
            "val_frac": VAL_FRAC,
            "target_mode": TARGET_MODE,
            "target_name": label_name,
            "model_family": MODEL_FAMILY,
            "n_params": overfit_info["n_params"],
            "samples_per_param": overfit_info["samples_per_param"],
            "overfit_risk": overfit_info["risk_level"],
        })

        # برو به فولد قبل‌تر (قدیمی‌تر): انتهای فولد بعدی = شروع فولد فعلی
        current_end = fold_start
        if current_end <= data_start:
            break

    report_df = pd.DataFrame(rows)

    # 🔹 ستون اضافی برای دیدن N/P به‌صورت صریح
    if len(report_df):
        report_df["np_ratio"] = report_df["samples_per_param"].astype(float)

    report_df.to_csv(wfv_csv_path, index=False, encoding=ENCODING)
    print(f"✅ [{config_name}] Walk-Forward report saved:", wfv_csv_path)
    print(report_df)

    # =========================
    # 4) Final train for deployment (fresh pipe)
    # =========================
    # مدل نهایی روی کل دیتای داخل Time Window (df_sorted) آموزش داده می‌شود
    final_train_mask = np.ones(len(df_sorted), dtype=bool)
    final_train_mask_np = final_train_mask
    X_final_np = X_all_np[final_train_mask_np]
    y_final = y_all[final_train_mask_np]

    final_pipe = clone(base_pipe)
    final_pipe.fit(X_final_np, y_final)

    # --- ریسک اوورفیت برای مدل نهایی ---
    clf_final_for_risk = final_pipe.named_steps["clf"]
    final_risk_info = estimate_overfit_risk(clf_final_for_risk, n_train_samples=len(y_final), verbose=True)

    # =========================
    # 5) Export to ONNX (xgb/lgbm/lr/mlp) + pre-graph
    # =========================
    engineered_len = N_RAW + N_DER  # در این دیتاست: N_DER=0 → engineered_len = N_RAW

    def strip_zipmap(core_model: onnx.ModelProto):
        graph = core_model.graph
        prob_out_name = None
        zipmap_idx = None
        for i, n in enumerate(graph.node):
            if n.op_type == "ZipMap" and len(n.input) == 1 and len(n.output) == 1:
                prezip = n.input[0]
                zipmap_out = n.output[0]
                prob_out_name = prezip
                zipmap_idx = i
                for out in graph.output:
                    if out.name == zipmap_out:
                        out.name = prezip
                break
        if zipmap_idx is not None:
            nodes = list(graph.node)
            del nodes[zipmap_idx]
            del graph.node[:]
            graph.node.extend(nodes)
            return core_model, prob_out_name
        for out in graph.output:
            t = out.type.tensor_type
            if t.elem_type == TensorProto.FLOAT:
                prob_out_name = out.name
                break
        if prob_out_name is None and len(graph.output):
            prob_out_name = graph.output[-1].name
        return core_model, prob_out_name

    def build_pregraph_plus_core(core_model: onnx.ModelProto, prob_name_hint=None):
        """
        دیتاست جدید:
        input  →  imputer (median)  →  core model
        هیچ فیچر مشتقی در ONNX ساخته نمی‌شود.
        """
        input_name = "input"  # ورودی خام با N_RAW ستون

        nodes = []
        inits = []

        # --- median imputer همانند SimpleImputer در پایپلاین ---
        imp = final_pipe.named_steps["imp"]
        medians = imp.statistics_.astype(np.float32)
        inits.append(numpy_helper.from_array(medians.reshape(1, engineered_len), name="med_row"))

        # X_nan → جایگزینی NaN با مدیان
        nodes.append(helper.make_node("IsNaN", [input_name], ["nan_mask"], name="imputer_isnan"))
        nodes.append(helper.make_node("Where", ["nan_mask", "med_row", input_name],
                                      ["engineered_in_imp"], name="imputer_where"))

        # ادغام با هستهٔ مدل
        core_model, core_prob_name = strip_zipmap(core_model)
        if prob_name_hint is not None:
            core_prob_name = prob_name_hint

        new_graph = helper.make_graph(
            nodes + list(core_model.graph.node),
            "full_graph_with_imputer_core",
            [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, N_RAW])],
            list(core_model.graph.output),
            initializer=inits + list(core_model.graph.initializer)
        )

        final_model = helper.make_model(new_graph)
        final_model.ir_version = core_model.ir_version

        del final_model.opset_import[:]
        for _imp in core_model.opset_import:
            oi = final_model.opset_import.add()
            oi.domain = _imp.domain
            oi.version = _imp.version

        return final_model, core_prob_name

    def add_profit_loss_projection(final_model: onnx.ModelProto, prob_output_name: str):
        final_out_name = "final_prob_1x4"
        for i in range(4):
            final_model.graph.initializer.add().CopyFrom(
                numpy_helper.from_array(np.array([i], dtype=np.int64), name=f"pp_idx{i}")
            )
        g0 = helper.make_node("Gather", [prob_output_name, "pp_idx0"], ["pp_p0"], axis=1, name="pp_g0")
        g1 = helper.make_node("Gather", [prob_output_name, "pp_idx1"], ["pp_p1"], axis=1, name="pp_g1")
        g2 = helper.make_node("Gather", [prob_output_name, "pp_idx2"], ["pp_p2"], axis=1, name="pp_g2")
        g3 = helper.make_node("Gather", [prob_output_name, "pp_idx3"], ["pp_p3"], axis=1, name="pp_g3")

        add01 = helper.make_node("Add", ["pp_p0", "pp_p1"], ["pp_p01"], name="pp_add01")  # p_profit
        add23 = helper.make_node("Add", ["pp_p2", "pp_p3"], ["pp_p23"], name="pp_add23")  # p_loss

        zero1 = helper.make_node("Sub", ["pp_p0", "pp_p0"], ["pp_zero1"], name="pp_zero1")
        zero2 = helper.make_node("Sub", ["pp_p2", "pp_p2"], ["pp_zero2"], name="pp_zero2")

        concat_pp = helper.make_node("Concat", ["pp_p01", "pp_zero1", "pp_p23", "pp_zero2"], ["pp_out"], axis=1,
                                     name="pp_concat")
        rename = helper.make_node("Identity", ["pp_out"], [final_out_name], name="pp_rename")

        final_model.graph.node.extend([g0, g1, g2, g3, add01, add23, zero1, zero2, concat_pp, rename])

        del final_model.graph.output[:]
        final_model.graph.output.add().CopyFrom(
            helper.make_tensor_value_info(final_out_name, TensorProto.FLOAT, [None, 4])
        )
        return final_model

    fam = MODEL_FAMILY.lower()
    onnx_built = False
    if fam in ("xgb", "lgbm", "lr", "mlp"):
        clf = final_pipe.named_steps["clf"]
        if fam == "xgb":
            core = convert_xgboost(
                clf,
                initial_types=[("engineered_in_imp", OXFloatTensorType([None, engineered_len]))],
                target_opset=15
            )
        elif fam == "lgbm":
            if not _HAS_CONVERT_LGBM:
                raise RuntimeError("onnxmltools.convert_lightgbm در دسترس نیست.")
            core = convert_lightgbm(
                clf,
                initial_types=[("engineered_in_imp", OXFloatTensorType([None, engineered_len]))],
                target_opset=15
            )
        else:
            core = convert_sklearn(
                clf,
                initial_types=[("engineered_in_imp", SKLFloatTensorType([None, engineered_len]))],
                target_opset=15
            )

        full, prob_out = build_pregraph_plus_core(core)
        full_final = add_profit_loss_projection(full, prob_out)

        onnx.save(full_final, onnx_path)
        print(f"✅ [{config_name}] Saved ONNX to:", onnx_path)
        print("✅ Saved feature JSON to:", FEATURE_JSON)
        onnx_built = True
    else:
        raise ValueError(f"Unknown MODEL_FAMILY: {MODEL_FAMILY}")

    # =========================
    # 6) Quick ONNX check
    # =========================
    if onnx_built:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        probe = X_all_np[-32:].astype(np.float32, copy=False)
        out = sess.run([out_name], {inp_name: probe})[0]
        print(f"[{config_name}] ONNX out shape:", out.shape)
        print("First row:", out[0])
        print("Zeros mean (cols 1 & 3):", float(out[:, 1].mean()), float(out[:, 3].mean()))

    # =========================
    # 7) Visualization of Precision / Recall / F1 (binary)
    # =========================
    if len(report_df):
        x_plot = report_df["fold_index"]  # 1 = جدیدترین فولد
        prec = report_df["precision_binary"]
        rec = report_df["recall_binary"]
        f1 = report_df["f1_binary"]

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, prec, marker="o", label=f"Precision ({TARGET_MODE})")
        plt.plot(x_plot, rec, marker="s", label=f"Recall ({TARGET_MODE})")
        plt.plot(x_plot, f1, marker="^", label=f"F1 ({TARGET_MODE})")

        plt.title(
            f"[{config_name}] Binary Precision / Recall / F1 over Folds – target={TARGET_CLASS if TARGET_MODE == 'single' else TARGET_GROUP} | model={MODEL_FAMILY}")
        plt.xlabel("Fold index (1 = newest)")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plt.savefig(plot_path, dpi=150)
        print(f"✅ [{config_name}] WFV metrics plot saved to:", plot_path)

        plt.show()

    # =========================
    # 8) Top-20 feature importance report
    # =========================
    print(f"\n===== [{config_name}] Top-20 feature importance (approx) =====")
    all_feature_names = RAW_FEATURES + DERIVED_NAMES
    importances = None

    clf_final = final_pipe.named_steps["clf"]
    fam = MODEL_FAMILY.lower()

    try:
        if fam in ("xgb", "lgbm") and hasattr(clf_final, "feature_importances_"):
            importances = np.asarray(clf_final.feature_importances_, dtype=float)
        elif fam == "lr" and hasattr(clf_final, "coef_"):
            coef = np.asarray(clf_final.coef_, dtype=float)
            if coef.ndim == 2:
                importances = np.mean(np.abs(coef), axis=0)
        elif fam == "mlp" and hasattr(clf_final, "coefs_") and len(clf_final.coefs_) > 0:
            w0 = np.asarray(clf_final.coefs_[0], dtype=float)  # (n_features, hidden)
            importances = np.mean(np.abs(w0), axis=1)
    except Exception as e:
        print("⚠️ Could not compute importances:", e)

    if importances is None or len(importances) != len(all_feature_names):
        print("⚠️ Feature importances not available or length mismatch; skipping Top-20 export.")
    else:
        imp_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": importances
        })
        imp_df_sorted = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
        top20 = imp_df_sorted.head(20).copy()
        top20["rank"] = np.arange(1, len(top20) + 1)

        print(top20.to_string(index=False))
        top20.to_csv(top20_path, index=False, encoding=ENCODING)
        print(f"✅ [{config_name}] Saved Top-20 feature importance to:", top20_path)

    # خلاصه‌ی عددی برای Grid
    mean_f1 = float(report_df["f1_binary"].mean()) if len(report_df) else float("nan")
    std_f1 = float(report_df["f1_binary"].std()) if len(report_df) else float("nan")
    min_f1 = float(report_df["f1_binary"].min()) if len(report_df) else float("nan")

    summary = {
        "config": config_name,
        "mean_f1_binary": mean_f1,
        "std_f1_binary": std_f1,
        "min_f1_binary": min_f1,
        "final_n_params": final_risk_info["n_params"],
        "final_samples_per_param": final_risk_info["samples_per_param"],
        "final_overfit_risk": final_risk_info["risk_level"],
    }
    print("\n===== SUMMARY for config", config_name, "=====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("============================================\n")

    return summary


# =========================
# اجرای نهایی: Single یا Grid
# =========================
if __name__ == '__main__':

    if USE_GRID and MODEL_FAMILY.lower() == "lgbm":
        summaries = []
        for cfg in LGBM_GRID:
            print("\n" + "=" * 80)
            print(f"▶ Running config: {cfg['name']}  params={cfg['params']}")
            s = run_for_config(config_name=cfg["name"], lgbm_params=cfg["params"])
            summaries.append(s)
        if summaries:
            grid_df = pd.DataFrame(summaries)
            print("\n===== GRID SUMMARY (all configs) =====")
            print(grid_df)
    else:
        # فقط یک بار با تنظیمات فعلی LGBM (یا مدل خانواده‌ی دیگر)
        _ = run_for_config(config_name="BASE", lgbm_params=None)
