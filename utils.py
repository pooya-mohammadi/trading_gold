from functools import partial

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
# ====== Base models ======
from xgboost import XGBClassifier
from settings import RAW_FEATURES

# ====== ONNX toolchains ======

try:
    from onnxmltools import convert_lightgbm

    _HAS_CONVERT_LGBM = True
except Exception:
    _HAS_CONVERT_LGBM = False

try:
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


def _idx(name: str, raw_features) -> int:
    try:
        return raw_features.index(name)
    except ValueError:
        raise ValueError(f"Required column '{name}' not found in RAW_FEATURES.")


# ---------- Feature builder (numpy, stateless)
def make_stateless_features(X_np: np.ndarray, raw_features) -> np.ndarray:
    """
    دیتاست جدید:
    - TimeFrame به کد دسته‌ای طبق tf_mapping تبدیل می‌شود.
    - بقیهٔ فیچرها بدون تغییر می‌مانند.
    ورودی: X_np با ستون‌های RAW_FEATURES
    خروجی: همان X به صورت float32 (با TimeFrame کد شده)
    """
    X = np.asarray(X_np, dtype=np.float32, order="C")

    # ====================================================
    # 1) تبدیل TimeFrame به کد دسته‌ای طبق مپینگ جدید
    # ====================================================
    tf_idx = _idx("TimeFrame", raw_features)
    tf_col = X[:, tf_idx].astype(np.float32)

    # NaN / inf → NaN نگه می‌داریم تا Imputer بعداً درست کند
    bad_mask = ~np.isfinite(tf_col)
    if bad_mask.any():
        tf_col[bad_mask] = np.nan

    tf_int = tf_col.copy()
    valid_mask = np.isfinite(tf_int)
    tf_int[valid_mask] = tf_int[valid_mask].astype(np.int64)

    # مپینگ جدید که خودت دادی
    tf_mapping = {
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 4.0,
        5: 5.0,
        6: 6.0,
        10: 7.0,
        12: 8.0,
        15: 9.0,
        20: 10.0,
        30: 11.0,
        16385: 12.0
    }

    # پیش‌فرض: NaN (اگر خارج از مپینگ باشد، به Imputer واگذار می‌شود)
    tf_cat = np.full_like(tf_col, np.nan, dtype=np.float32)

    for v, code in tf_mapping.items():
        match_mask = (tf_int == v) & valid_mask
        tf_cat[match_mask] = code

    # جایگزینی ستون TimeFrame با نسخهٔ دسته‌ای (کدها یا NaN)
    X[:, tf_idx] = tf_cat

    return X


# ---------- سtep: enforce NumPy(float32) at pipeline input
def _to_numpy32(X):
    return np.asarray(X, dtype=np.float32, order="C")


# =========================
# تخمین تعداد پارامتر و ریسک اوورفیت
# =========================
def _count_params_linear_or_logistic(model):
    if not hasattr(model, "coef_"):
        raise ValueError("Model has no coef_. Make sure it is already fitted.")
    coef = np.asarray(model.coef_)
    n_params = coef.size
    if hasattr(model, "intercept_"):
        n_params += np.asarray(model.intercept_).size
    return int(n_params)


def _count_params_mlp(model):
    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise ValueError("MLP model has no coefs_/intercepts_. Make sure it is fitted.")
    n_params = 0
    for w in model.coefs_:
        n_params += np.asarray(w).size
    for b in model.intercepts_:
        n_params += np.asarray(b).size
    return int(n_params)


def _count_params_tree_ensemble(model):
    n_estimators = getattr(model, "n_estimators", None)
    if n_estimators is None:
        raise ValueError("Tree model has no n_estimators attribute (unexpected).")

    num_leaves = getattr(model, "num_leaves", None)
    max_depth = getattr(model, "max_depth", None)

    if num_leaves is not None and num_leaves > 0:
        nodes_per_tree = float(num_leaves)
    else:
        if max_depth is not None and max_depth > 0:
            nodes_per_tree = float(2 ** (max_depth + 1) - 1)
        else:
            nodes_per_tree = 256.0  # تخمین محافظه‌کارانه

    n_params = int(n_estimators * nodes_per_tree)
    return n_params


def estimate_model_params(model):
    if isinstance(model, LogisticRegression):
        return _count_params_linear_or_logistic(model)
    if isinstance(model, MLPClassifier):
        return _count_params_mlp(model)
    if isinstance(model, XGBClassifier):
        return _count_params_tree_ensemble(model)
    if _HAS_LGBM and isinstance(model, lgb.LGBMClassifier):
        return _count_params_tree_ensemble(model)
    raise ValueError(f"Cannot estimate parameters for model type: {model.__class__.__name__}")


def categorize_overfit_risk(n_samples, n_params):
    if n_params <= 0:
        return "unknown", np.inf

    ratio = float(n_samples) / float(n_params)

    if ratio < 5:
        level = "🚨 VERY HIGH overfit risk"
    elif ratio < 10:
        level = "⚠ HIGH overfit risk"
    elif ratio < 50:
        level = "🟡 MODERATE overfit risk"
    elif ratio < 200:
        level = "🟢 LOW overfit risk"
    else:
        level = "✅ VERY LOW overfit risk"

    return level, ratio


def estimate_overfit_risk(model, n_train_samples, verbose=True):
    n_params = estimate_model_params(model)
    risk_level, ratio = categorize_overfit_risk(n_train_samples, n_params)
    info = {
        "n_params": int(n_params),
        "n_train_samples": int(n_train_samples),
        "samples_per_param": float(ratio),
        "risk_level": risk_level,
    }
    if verbose:
        print("===== Overfit risk estimation =====")
        print(f"Model: {model.__class__.__name__}")
        print(f"Train samples (N): {n_train_samples}")
        print(f"Estimated parameters (P): {n_params}")
        print(f"Samples per parameter (N/P): {ratio:,.3f}")
        print(f"Risk level: {risk_level}")
    return info


def make_classifier(model_family: str, val_frac, num_classes=4, current_lgbm_params: dict = None):
    fam = model_family.lower()
    if fam == "xgb":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    elif fam == "lgbm":
        if not _HAS_LGBM:
            raise RuntimeError("LightGBM نصب نیست. `pip install lightgbm`")
        base_params = dict(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=700,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            force_col_wise=True,  # حذف سربار تست col-wise
            verbosity=-1  # لاگ کمتر
        )
        if current_lgbm_params is not None:
            base_params.update(current_lgbm_params)
        return lgb.LGBMClassifier(**base_params)
    elif fam == "lr":
        return LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="lbfgs",
            max_iter=5000,
            tol=1e-4
        )
    elif fam == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=val_frac,
            tol=1e-4,
            shuffle=True,
            random_state=42,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown MODEL_FAMILY: {model_family}")


# ترنسفورمر فیچرها
feat_builder = FunctionTransformer(partial(make_stateless_features, raw_features=RAW_FEATURES), validate=False)
to_numpy = FunctionTransformer(_to_numpy32, validate=False)


# Base pipeline (clone per run)
def make_base_pipeline(model_family, val_frac, lgbm_params: dict):
    # clf = make_classifier(num_classes=4, model_family=model_family, val_frac=val_frac, current_lgbm_params=lgbm_params)
    from tabpfn.classifier import TabPFNClassifier
    clf = TabPFNClassifier()
    fam = model_family.lower()
    steps = [
        ("to_np", to_numpy),  # تضمین NumPy
        ("feat", feat_builder),
        ("imp", SimpleImputer(strategy="median"))
    ]
    if fam in ("lr", "mlp"):
        steps.append(("scaler", StandardScaler(with_mean=True)))  # اسکیلینگ واقعی
    steps.append(("clf", clf))
    return Pipeline(steps=steps)
