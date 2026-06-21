import os
import time
import json
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)

# ============================================================
# Optional advanced tabular model packages
# ============================================================

SKIPPED_MODELS = []

LIGHTGBM_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    LGBMClassifier = None
    SKIPPED_MODELS.append({"model": "LightGBM", "reason": f"Không import được lightgbm: {e}"})

XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBClassifier = None
    SKIPPED_MODELS.append({"model": "XGBoost", "reason": f"Không import được xgboost: {e}"})

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception as e:
    CatBoostClassifier = None
    SKIPPED_MODELS.append({"model": "CatBoost", "reason": f"Không import được catboost: {e}"})

TABM_AVAILABLE = False
try:
    import torch
    from torch import nn
    from tabm import TabM
    TABM_AVAILABLE = True
except Exception as e:
    torch = None
    nn = None
    TabM = None
    SKIPPED_MODELS.append({"model": "TabM 2025", "reason": f"Không import được tabm/torch: {e}"})

# TabPFN đã thử ở bản 07 nhưng dễ bị chặn bởi license/login và lỗi socket Windows.
# Để tránh chương trình treo khi chạy batch, mặc định KHÔNG chạy TabPFN.
ENABLE_TABPFN = False
TABPFN_AVAILABLE = False
try:
    if ENABLE_TABPFN:
        from tabpfn import TabPFNClassifier
        TABPFN_AVAILABLE = True
    else:
        TabPFNClassifier = None
        SKIPPED_MODELS.append({"model": "TabPFN", "reason": "Tắt mặc định để tránh yêu cầu đăng nhập/license và lỗi socket Windows; có thể bật ENABLE_TABPFN=True nếu muốn thử riêng."})
except Exception as e:
    TabPFNClassifier = None
    SKIPPED_MODELS.append({"model": "TabPFN", "reason": f"Không import/khởi tạo được TabPFN: {e}"})


class DenseTransformer(BaseEstimator):
    """Convert sparse matrices from OneHotEncoder/ColumnTransformer to dense arrays."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


class TabMClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-compatible wrapper for the official tabm package.

    This wrapper feeds the already-preprocessed dense tabular feature matrix to TabM
    as numerical features only. It is intentionally small and experimental: use it
    for benchmarking, not as a guaranteed production implementation.
    """
    def __init__(
        self,
        max_epochs=80,
        lr=1e-3,
        batch_size=256,
        weight_decay=1e-4,
        random_state=42,
        verbose=0,
        device="cpu"
    ):
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

    def fit(self, X, y):
        if not TABM_AVAILABLE:
            raise ImportError("tabm/torch chưa được cài. Chạy: pip install tabm")

        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model_ = TabM.make(
            n_num_features=self.n_features_in_,
            cat_cardinalities=[],
            d_out=1,
        ).to(self.device)

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # class imbalance handling similar to class_weight='balanced'
        pos = float(y_tensor.sum().item())
        neg = float(len(y_tensor) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n = X_tensor.shape[0]
        indices = np.arange(n)

        self.model_.train()
        for epoch in range(self.max_epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                xb = X_tensor[batch_idx]
                yb = y_tensor[batch_idx]

                optimizer.zero_grad()
                logits = self.model_(xb, None).squeeze(-1).mean(dim=1, keepdim=True)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_idx)

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"TabM epoch {epoch+1}/{self.max_epochs}, loss={epoch_loss/n:.4f}")

        return self

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor, None).squeeze(-1).mean(dim=1)
            proba = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs_benchmark08"
FIGURE_DIR = "figures_benchmark08"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42
N_RANDOM_ITER = 8
BENCHMARK_NAME = "08 extended benchmark"


# ============================================================
# 1. Tìm file dữ liệu
# ============================================================

def find_file(filename):
    search_folders = [
        ".",
        "data",
        "blinkit_data2",
        "..",
        os.path.join("..", "data"),
        os.path.join("..", "blinkit_data2")
    ]

    for folder in search_folders:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Không tìm thấy file {filename}")


# ============================================================
# 2. Đọc dữ liệu
# ============================================================

def load_data():
    orders_path = find_file("blinkit_orders_clean.csv")
    order_items_path = find_file("blinkit_order_items.csv")
    products_path = find_file("blinkit_products.csv")

    orders = pd.read_csv(orders_path)
    order_items = pd.read_csv(order_items_path)
    products = pd.read_csv(products_path)

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")

    for col in ["quantity", "unit_price"]:
        order_items[col] = pd.to_numeric(
            order_items[col],
            errors="coerce"
        ).fillna(0)

    for col in ["price", "mrp", "margin_percentage", "shelf_life_days"]:
        if col in products.columns:
            products[col] = pd.to_numeric(
                products[col],
                errors="coerce"
            ).fillna(0)

    print("Đã đọc dữ liệu:")
    print(f"Orders clean: {orders.shape} - {orders_path}")
    print(f"Order items:  {order_items.shape} - {order_items_path}")
    print(f"Products:     {products.shape} - {products_path}")

    return orders, order_items, products


# ============================================================
# 3. Kiểm tra chất lượng dữ liệu
# ============================================================

def create_data_quality_summary(orders, order_items, products):
    rows = []

    for name, df in [
        ("orders_clean", orders),
        ("order_items", order_items),
        ("products", products)
    ]:
        rows.append({
            "table": name,
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "missing_cells": int(df.isna().sum().sum()),
            "duplicated_rows": int(df.duplicated().sum())
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "final_data_quality_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# ============================================================
# 4. Loại tháng không đầy đủ
# ============================================================

def get_complete_months(orders):
    orders = orders.dropna(subset=["order_date"]).copy()
    orders["order_month"] = orders["order_date"].dt.to_period("M")

    min_date = orders["order_date"].min()
    max_date = orders["order_date"].max()

    all_months = pd.period_range(
        min_date.to_period("M"),
        max_date.to_period("M"),
        freq="M"
    )

    incomplete_months = set()

    # Tháng đầu không đầy đủ nếu dữ liệu không bắt đầu từ ngày 1
    if min_date.day > 1:
        incomplete_months.add(min_date.to_period("M"))

    # Tháng cuối không đầy đủ nếu dữ liệu không kết thúc ở ngày cuối tháng
    if max_date.day < max_date.days_in_month:
        incomplete_months.add(max_date.to_period("M"))

    complete_months = [
        month for month in all_months
        if month not in incomplete_months
    ]

    coverage = (
        orders
        .groupby("order_month")
        .agg(
            n_orders=("order_id", "count"),
            first_date=("order_date", "min"),
            last_date=("order_date", "max")
        )
        .reset_index()
    )

    coverage["order_month"] = coverage["order_month"].astype(str)
    coverage["is_used_for_model"] = coverage["order_month"].isin(
        [str(m) for m in complete_months]
    )

    coverage.to_csv(
        os.path.join(OUTPUT_DIR, "final_month_coverage_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nKiểm tra tháng dữ liệu:")
    print(f"Ngày đầu: {min_date}")
    print(f"Ngày cuối: {max_date}")

    if incomplete_months:
        print(
            "Loại tháng không đầy đủ:",
            ", ".join(str(m) for m in sorted(incomplete_months))
        )
    else:
        print("Không phát hiện tháng đầu/cuối bị thiếu.")

    print(
        "Dải tháng dùng cho mô hình:",
        f"{complete_months[0]} đến {complete_months[-1]}"
    )

    return complete_months, coverage


# ============================================================
# 5. Tạo sales monthly panel
# ============================================================

def build_monthly_sales(orders, order_items, products, complete_months):
    order_time = orders[["order_id", "order_date"]].copy()

    sales = order_items.merge(order_time, on="order_id", how="left")
    sales = sales.dropna(subset=["order_date"])

    sales["sales_amount"] = sales["quantity"] * sales["unit_price"]
    sales["order_month"] = sales["order_date"].dt.to_period("M")

    sales = sales[sales["order_month"].isin(complete_months)].copy()

    monthly_sales = (
        sales
        .groupby(["product_id", "order_month"])
        .agg(
            monthly_quantity=("quantity", "sum"),
            monthly_revenue=("sales_amount", "sum"),
            monthly_order_lines=("order_id", "count"),
            monthly_orders=("order_id", "nunique")
        )
        .reset_index()
    )

    product_ids = products["product_id"].drop_duplicates()

    full_index = pd.MultiIndex.from_product(
        [product_ids, complete_months],
        names=["product_id", "order_month"]
    )

    monthly_sales = (
        monthly_sales
        .set_index(["product_id", "order_month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    monthly_sales_for_csv = monthly_sales.copy()
    monthly_sales_for_csv["order_month"] = monthly_sales_for_csv["order_month"].astype(str)

    monthly_sales_for_csv.to_csv(
        os.path.join(OUTPUT_DIR, "final_monthly_product_sales_panel.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return monthly_sales


# ============================================================
# 6. Tạo supervised dataset theo rolling cutoff
# ============================================================

def safe_divide(a, b):
    return np.where(b != 0, a / b, 0)


def create_rolling_supervised_dataset(
    monthly_sales,
    products,
    min_history_months=6,
    horizon_months=3,
    positive_fraction=0.25
):
    rows = []

    all_months = sorted(monthly_sales["order_month"].unique())
    all_products = products["product_id"].drop_duplicates().tolist()

    for cutoff_idx in range(min_history_months, len(all_months) - horizon_months + 1):
        history_months = all_months[:cutoff_idx]
        future_months = all_months[cutoff_idx: cutoff_idx + horizon_months]

        cutoff_month = str(all_months[cutoff_idx - 1])
        target_start_month = str(all_months[cutoff_idx])
        target_end_month = str(all_months[cutoff_idx + horizon_months - 1])

        hist = monthly_sales[monthly_sales["order_month"].isin(history_months)]
        fut = monthly_sales[monthly_sales["order_month"].isin(future_months)]

        hist_agg = (
            hist
            .groupby("product_id")
            .agg(
                hist_total_quantity=("monthly_quantity", "sum"),
                hist_total_revenue=("monthly_revenue", "sum"),
                hist_total_orders=("monthly_orders", "sum"),
                hist_mean_monthly_quantity=("monthly_quantity", "mean"),
                hist_std_monthly_quantity=("monthly_quantity", "std"),
                hist_max_monthly_quantity=("monthly_quantity", "max"),
                hist_mean_monthly_revenue=("monthly_revenue", "mean"),
                hist_std_monthly_revenue=("monthly_revenue", "std"),
                hist_active_months=("monthly_quantity", lambda x: int((x > 0).sum()))
            )
            .reset_index()
        )

        fut_agg = (
            fut
            .groupby("product_id")
            .agg(
                future_3m_quantity=("monthly_quantity", "sum"),
                future_3m_revenue=("monthly_revenue", "sum"),
                future_3m_orders=("monthly_orders", "sum")
            )
            .reset_index()
        )

        fold_df = pd.DataFrame({"product_id": all_products})
        fold_df = fold_df.merge(hist_agg, on="product_id", how="left")
        fold_df = fold_df.merge(fut_agg, on="product_id", how="left")
        fold_df = fold_df.merge(products, on="product_id", how="left")

        numeric_cols = [
            "hist_total_quantity",
            "hist_total_revenue",
            "hist_total_orders",
            "hist_mean_monthly_quantity",
            "hist_std_monthly_quantity",
            "hist_max_monthly_quantity",
            "hist_mean_monthly_revenue",
            "hist_std_monthly_revenue",
            "hist_active_months",
            "future_3m_quantity",
            "future_3m_revenue",
            "future_3m_orders",
            "price",
            "mrp",
            "margin_percentage",
            "shelf_life_days"
        ]

        for col in numeric_cols:
            if col in fold_df.columns:
                fold_df[col] = pd.to_numeric(
                    fold_df[col],
                    errors="coerce"
                ).fillna(0)

        fold_df["hist_demand_cv"] = safe_divide(
            fold_df["hist_std_monthly_quantity"],
            fold_df["hist_mean_monthly_quantity"]
        )

        fold_df["hist_revenue_cv"] = safe_divide(
            fold_df["hist_std_monthly_revenue"],
            fold_df["hist_mean_monthly_revenue"]
        )

        fold_df["hist_sales_frequency"] = safe_divide(
            fold_df["hist_active_months"],
            len(history_months)
        )

        fold_df["estimated_hist_profit"] = (
            fold_df["hist_total_revenue"] * fold_df["margin_percentage"] / 100
        )

        fold_df["discount_rate"] = safe_divide(
            fold_df["mrp"] - fold_df["price"],
            fold_df["mrp"]
        )

        fold_df["discount_rate"] = np.clip(fold_df["discount_rate"], 0, 1)

        fold_df = fold_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Gán nhãn top 25% sản phẩm theo future_3m_revenue trong từng cutoff.
        # Cách này giữ tỷ lệ lớp dương ổn định và tránh lỗi do nhiều giá trị bằng nhau ở quantile.
        fold_df["future_revenue_rank"] = fold_df["future_3m_revenue"].rank(
            method="first",
            ascending=False
        )

        n_positive = int(np.ceil(len(fold_df) * positive_fraction))

        fold_df["high_future_performance"] = (
            fold_df["future_revenue_rank"] <= n_positive
        ).astype(int)

        fold_df["cutoff_month"] = cutoff_month
        fold_df["target_start_month"] = target_start_month
        fold_df["target_end_month"] = target_end_month
        fold_df["fold_id"] = cutoff_idx

        rows.append(fold_df)

    supervised = pd.concat(rows, ignore_index=True)
    supervised = supervised.replace([np.inf, -np.inf], np.nan).fillna(0)

    supervised.to_csv(
        os.path.join(OUTPUT_DIR, "final_supervised_product_performance_dataset.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    label_distribution = (
        supervised
        .groupby(["fold_id", "cutoff_month", "target_start_month", "target_end_month"])
        .agg(
            n_samples=("product_id", "count"),
            n_positive=("high_future_performance", "sum"),
            positive_rate=("high_future_performance", "mean"),
            avg_future_revenue=("future_3m_revenue", "mean")
        )
        .reset_index()
    )

    label_distribution.to_csv(
        os.path.join(OUTPUT_DIR, "final_label_distribution_by_fold.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return supervised, label_distribution


# ============================================================
# 7. Train/test split theo thời gian
# ============================================================

def temporal_train_test_split(df, n_test_folds=3):
    fold_ids = sorted(df["fold_id"].unique())

    test_folds = fold_ids[-n_test_folds:]
    train_folds = fold_ids[:-n_test_folds]

    train_df = df[df["fold_id"].isin(train_folds)].copy().reset_index(drop=True)
    test_df = df[df["fold_id"].isin(test_folds)].copy().reset_index(drop=True)

    return train_df, test_df, train_folds, test_folds


# ============================================================
# 8. Features và preprocessing
# ============================================================

def get_features():
    numeric_features = [
        "price",
        "mrp",
        "margin_percentage",
        "shelf_life_days",
        "discount_rate",
        "hist_total_quantity",
        "hist_total_revenue",
        "hist_total_orders",
        "hist_mean_monthly_quantity",
        "hist_std_monthly_quantity",
        "hist_max_monthly_quantity",
        "hist_mean_monthly_revenue",
        "hist_std_monthly_revenue",
        "hist_active_months",
        "hist_demand_cv",
        "hist_revenue_cv",
        "hist_sales_frequency",
        "estimated_hist_profit"
    ]

    categorical_features = [
        "category"
    ]

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )


# ============================================================
# 9. Models và hyperparameter search
# ============================================================

def get_classification_models(preprocessor):
    models = {
        "Logistic Regression Balanced": {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("model", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "model__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
            }
        },

        "Linear SVM Balanced": {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("model", SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "model__C": [0.05, 0.1, 0.3, 1.0, 3.0]
            }
        },

        "Decision Tree Balanced": {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("model", DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "model__max_depth": [3, 5, 8, 12],
                "model__min_samples_leaf": [5, 10, 20],
                "model__min_samples_split": [10, 20, 40]
            }
        },

        "Random Forest Balanced": {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1
                ))
            ]),
            "params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_leaf": [3, 5, 10],
                "model__max_features": ["sqrt", "log2", None]
            }
        },

        "Extra Trees Balanced": {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("model", ExtraTreesClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1
                ))
            ]),
            "params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_leaf": [3, 5, 10],
                "model__max_features": ["sqrt", "log2", None]
            }
        },

        "Gradient Boosting": {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("model", GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.8, 1.0]
            }
        },

        "HistGradientBoosting": {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE
                ))
            ]),
            "params": {
                "model__max_iter": [100, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.01, 0.1]
            }
        }
    }

    # ------------------------------------------------------------
    # Additional classical / industry-strength tabular models
    # ------------------------------------------------------------

    models["Gaussian Naive Bayes"] = {
        "estimator": Pipeline([
            ("preprocess", preprocessor),
            ("to_dense", DenseTransformer()),
            ("model", GaussianNB())
        ]),
        "params": {
            "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }

    models["KNN Classifier"] = {
        "estimator": Pipeline([
            ("preprocess", preprocessor),
            ("to_dense", DenseTransformer()),
            ("model", KNeighborsClassifier())
        ]),
        "params": {
            "model__n_neighbors": [5, 9, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
    }

    models["MLP Classifier"] = {
        "estimator": Pipeline([
            ("preprocess", preprocessor),
            ("to_dense", DenseTransformer()),
            ("model", MLPClassifier(
                max_iter=600,
                early_stopping=True,
                random_state=RANDOM_STATE
            ))
        ]),
        "params": {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [1e-3, 3e-4]
        }
    }

    if LIGHTGBM_AVAILABLE:
        models["LightGBM Balanced"] = {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("to_dense", DenseTransformer()),
                ("model", LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=-1
                ))
            ]),
            "params": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
                "model__max_depth": [-1, 5, 8],
                "model__min_child_samples": [10, 20, 40]
            }
        }

    if XGBOOST_AVAILABLE:
        models["XGBoost Balanced"] = {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("to_dense", DenseTransformer()),
                ("model", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist"
                ))
            ]),
            "params": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 4, 6],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        }

    if CATBOOST_AVAILABLE:
        models["CatBoost Balanced"] = {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("to_dense", DenseTransformer()),
                ("model", CatBoostClassifier(
                    loss_function="Logloss",
                    auto_class_weights="Balanced",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    thread_count=1
                ))
            ]),
            "params": {
                "model__iterations": [100, 200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__depth": [4, 6, 8],
                "model__l2_leaf_reg": [1, 3, 5]
            }
        }

    if TABM_AVAILABLE:
        models["TabM 2025 Experimental"] = {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("to_dense", DenseTransformer()),
                ("model", TabMClassifierWrapper(random_state=RANDOM_STATE, device="cpu"))
            ]),
            "params": {
                "model__max_epochs": [60, 100],
                "model__lr": [1e-3, 3e-4],
                "model__weight_decay": [1e-4, 1e-3]
            }
        }

    if TABPFN_AVAILABLE:
        models["TabPFN Optional"] = {
            "estimator": Pipeline([
                ("preprocess", preprocessor),
                ("to_dense", DenseTransformer()),
                ("model", TabPFNClassifier(device="cpu"))
            ]),
            "params": {}
        }

    return models


def count_param_combinations(param_grid):
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def clean_params(params):
    cleaned = {}

    for key, value in params.items():
        if isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value

    return cleaned


def create_temporal_cv_splits(train_df, n_cv_folds=3):
    fold_ids = sorted(train_df["fold_id"].unique())

    if len(fold_ids) <= 2:
        return None

    validation_folds = fold_ids[-n_cv_folds:]
    cv_splits = []

    for val_fold in validation_folds:
        train_idx = train_df.index[train_df["fold_id"] < val_fold].to_numpy()
        val_idx = train_df.index[train_df["fold_id"] == val_fold].to_numpy()

        if len(train_idx) > 0 and len(val_idx) > 0:
            cv_splits.append((train_idx, val_idx))

    return cv_splits


# ============================================================
# 10. Threshold tuning
# ============================================================

def get_positive_probability(model, X):
    base_model = model.named_steps["model"]

    if hasattr(base_model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(base_model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    return model.predict(X)


def find_best_threshold(y_true, y_proba):
    best_threshold = 0.5
    best_f1 = -1
    best_recall = -1
    best_precision = -1

    thresholds = np.arange(0.10, 0.91, 0.01)

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        cur_f1 = f1_score(y_true, y_pred, zero_division=0)
        cur_recall = recall_score(y_true, y_pred, zero_division=0)
        cur_precision = precision_score(y_true, y_pred, zero_division=0)

        if (
            cur_f1 > best_f1
            or (
                cur_f1 == best_f1
                and cur_recall > best_recall
            )
            or (
                cur_f1 == best_f1
                and cur_recall == best_recall
                and cur_precision > best_precision
            )
        ):
            best_f1 = cur_f1
            best_recall = cur_recall
            best_precision = cur_precision
            best_threshold = threshold

    return best_threshold, best_f1, best_precision, best_recall


def select_threshold_on_validation(model, X_train, y_train, train_df):
    fold_ids = sorted(train_df["fold_id"].unique())

    if len(fold_ids) < 2:
        return 0.5, np.nan, np.nan, np.nan

    validation_fold = fold_ids[-1]

    inner_idx = train_df.index[train_df["fold_id"] < validation_fold].to_numpy()
    val_idx = train_df.index[train_df["fold_id"] == validation_fold].to_numpy()

    if len(inner_idx) == 0 or len(val_idx) == 0:
        return 0.5, np.nan, np.nan, np.nan

    threshold_model = clone(model)
    threshold_model.fit(X_train.iloc[inner_idx], y_train.iloc[inner_idx])

    val_proba = get_positive_probability(
        threshold_model,
        X_train.iloc[val_idx]
    )

    threshold, val_f1, val_precision, val_recall = find_best_threshold(
        y_train.iloc[val_idx],
        val_proba
    )

    return threshold, val_f1, val_precision, val_recall


# ============================================================
# 11. Ranking metrics
# ============================================================

def ranking_metrics_at_fraction(y_true, y_proba, fraction):
    ranking_df = pd.DataFrame({
        "actual": y_true.values,
        "probability": y_proba
    }).sort_values("probability", ascending=False)

    k = max(1, int(np.ceil(len(ranking_df) * fraction)))
    top_k = ranking_df.head(k)

    precision_at_k = top_k["actual"].mean()

    total_positive = ranking_df["actual"].sum()
    recall_at_k = top_k["actual"].sum() / total_positive if total_positive > 0 else 0

    base_rate = ranking_df["actual"].mean()
    lift_at_k = precision_at_k / base_rate if base_rate > 0 else 0

    return k, precision_at_k, recall_at_k, lift_at_k


# ============================================================
# 12. Fit + evaluate
# ============================================================

def fit_model_with_tuning(
    model_name,
    estimator,
    param_grid,
    X_train,
    y_train,
    train_df,
    cv_splits
):
    start_train = time.time()

    if param_grid and cv_splits:
        n_iter = min(N_RANDOM_ITER, count_param_combinations(param_grid))

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="f1",
            cv=cv_splits,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True
        )

        search.fit(X_train, y_train)

        fitted_model = search.best_estimator_
        best_params = clean_params(search.best_params_)
        best_cv_score = float(search.best_score_)

    else:
        fitted_model = clone(estimator)
        fitted_model.fit(X_train, y_train)
        best_params = {}
        best_cv_score = np.nan

    threshold, val_f1, val_precision, val_recall = select_threshold_on_validation(
        fitted_model,
        X_train,
        y_train,
        train_df
    )

    train_time = time.time() - start_train

    return {
        "model": fitted_model,
        "threshold": threshold,
        "best_params": best_params,
        "best_cv_f1": best_cv_score,
        "validation_threshold_f1": val_f1,
        "validation_threshold_precision": val_precision,
        "validation_threshold_recall": val_recall,
        "train_time_seconds": train_time
    }


def evaluate_classifier(
    model_name,
    fitted_model,
    threshold,
    best_params,
    best_cv_f1,
    validation_threshold_f1,
    validation_threshold_precision,
    validation_threshold_recall,
    train_time_seconds,
    X_test,
    y_test
):
    start_pred = time.time()

    y_proba = get_positive_probability(fitted_model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    predict_time_seconds = time.time() - start_pred

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    k10, p10, r10, lift10 = ranking_metrics_at_fraction(y_test, y_proba, 0.10)
    k25, p25, r25, lift25 = ranking_metrics_at_fraction(y_test, y_proba, 0.25)

    result = {
        "model": model_name,
        "threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "PR_AUC": average_precision_score(y_test, y_proba),
        "Brier_Score": brier_score_loss(y_test, y_proba),
        "Top10_K": k10,
        "Precision_at_Top10pct": p10,
        "Recall_at_Top10pct": r10,
        "Lift_at_Top10pct": lift10,
        "Top25_K": k25,
        "Precision_at_Top25pct": p25,
        "Recall_at_Top25pct": r25,
        "Lift_at_Top25pct": lift25,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "best_cv_f1": best_cv_f1,
        "validation_threshold_f1": validation_threshold_f1,
        "validation_threshold_precision": validation_threshold_precision,
        "validation_threshold_recall": validation_threshold_recall,
        "best_params": json.dumps(best_params, ensure_ascii=False),
        "train_time_seconds": train_time_seconds,
        "predict_time_seconds": predict_time_seconds,
        "predict_time_ms": predict_time_seconds * 1000
    }

    pred_df = pd.DataFrame({
        "model": model_name,
        "actual": y_test.values,
        "predicted": y_pred,
        "probability_high_performance": y_proba
    })

    return result, pred_df


# ============================================================
# 13. Figures
# ============================================================

def plot_label_distribution(label_distribution):
    plt.figure(figsize=(10, 5))
    plt.bar(
        label_distribution["cutoff_month"].astype(str),
        label_distribution["positive_rate"]
    )
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Positive rate")
    plt.xlabel("Cutoff month")
    plt.title("Tỷ lệ sản phẩm hiệu quả cao theo từng cutoff")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "final_label_distribution_by_fold.png"),
        dpi=300
    )
    plt.close()


def plot_classification_results(results_df):
    plot_df = results_df.sort_values("F1", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["model"], plot_df["F1"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("F1-score")
    plt.title("So sánh mô hình phân lớp theo F1-score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "final_classification_f1_comparison.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["model"], plot_df["ROC_AUC"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("ROC-AUC")
    plt.title("So sánh mô hình phân lớp theo ROC-AUC")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "final_classification_roc_auc_comparison.png"),
        dpi=300
    )
    plt.close()

    best = plot_df.iloc[0]

    cm = np.array([
        [best["TN"], best["FP"]],
        [best["FN"], best["TP"]]
    ])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.title(f"Confusion Matrix - {best['model']}")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "final_best_confusion_matrix.png"),
        dpi=300
    )
    plt.close()


def plot_best_model_curves(y_test, best_proba, best_model_name):
    fpr, tpr, _ = roc_curve(y_test, best_proba)
    precision, recall, _ = precision_recall_curve(y_test, best_proba)

    roc_auc = roc_auc_score(y_test, best_proba)
    pr_auc = average_precision_score(y_test, best_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {best_model_name} (AUC={roc_auc:.4f})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "final_best_model_roc_curve.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {best_model_name} (AP={pr_auc:.4f})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "final_best_model_pr_curve.png"),
        dpi=300
    )
    plt.close()


def save_feature_importance(best_model, best_model_name):
    base_model = best_model.named_steps["model"]
    preprocessor = best_model.named_steps["preprocess"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(base_model, "feature_importances_"):
        importance = base_model.feature_importances_
    elif hasattr(base_model, "coef_"):
        importance = np.abs(base_model.coef_).ravel()
    else:
        return None

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv(
        os.path.join(OUTPUT_DIR, "final_best_model_feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    top_features = feature_importance.head(20).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 20 feature importance - {best_model_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "final_best_model_feature_importance.png"),
        dpi=300
    )
    plt.close()

    return feature_importance


# ============================================================
# 14. Output tables
# ============================================================

def create_recommendation_and_error_tables(
    test_df,
    X_test,
    y_test,
    best_model,
    best_model_name,
    best_threshold
):
    best_proba = get_positive_probability(best_model, X_test)
    best_pred = (best_proba >= best_threshold).astype(int)

    recommendation_table = test_df[[
        "product_id",
        "product_name",
        "category",
        "brand",
        "cutoff_month",
        "target_start_month",
        "target_end_month",
        "future_3m_revenue",
        "future_3m_quantity",
        "future_revenue_rank",
        "high_future_performance",
        "hist_total_quantity",
        "hist_total_revenue",
        "hist_mean_monthly_quantity",
        "hist_demand_cv",
        "hist_sales_frequency",
        "price",
        "mrp",
        "margin_percentage",
        "discount_rate",
        "shelf_life_days"
    ]].copy()

    recommendation_table["best_model"] = best_model_name
    recommendation_table["classification_threshold"] = best_threshold
    recommendation_table["predicted_high_performance"] = best_pred
    recommendation_table["probability_high_performance"] = best_proba

    recommendation_table = recommendation_table.sort_values(
        "probability_high_performance",
        ascending=False
    ).reset_index(drop=True)

    recommendation_table.to_csv(
        os.path.join(OUTPUT_DIR, "final_future_performance_recommendation_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    latest_cutoff = recommendation_table["cutoff_month"].max()

    latest_recommendation_table = (
        recommendation_table[
            recommendation_table["cutoff_month"] == latest_cutoff
        ]
        .sort_values("probability_high_performance", ascending=False)
        .reset_index(drop=True)
    )

    latest_recommendation_table.to_csv(
        os.path.join(OUTPUT_DIR, "final_latest_recommendation_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    false_positive = recommendation_table[
        (recommendation_table["predicted_high_performance"] == 1)
        & (recommendation_table["high_future_performance"] == 0)
    ].sort_values("probability_high_performance", ascending=False)

    false_negative = recommendation_table[
        (recommendation_table["predicted_high_performance"] == 0)
        & (recommendation_table["high_future_performance"] == 1)
    ].sort_values("future_3m_revenue", ascending=False)

    true_positive = recommendation_table[
        (recommendation_table["predicted_high_performance"] == 1)
        & (recommendation_table["high_future_performance"] == 1)
    ].sort_values("probability_high_performance", ascending=False)

    true_negative = recommendation_table[
        (recommendation_table["predicted_high_performance"] == 0)
        & (recommendation_table["high_future_performance"] == 0)
    ].sort_values("probability_high_performance", ascending=True)

    false_positive.to_csv(
        os.path.join(OUTPUT_DIR, "final_false_positive_cases.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    false_negative.to_csv(
        os.path.join(OUTPUT_DIR, "final_false_negative_cases.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    true_positive.to_csv(
        os.path.join(OUTPUT_DIR, "final_true_positive_cases.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    true_negative.to_csv(
        os.path.join(OUTPUT_DIR, "final_true_negative_cases.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return (
        recommendation_table,
        latest_recommendation_table,
        false_positive,
        false_negative,
        true_positive,
        true_negative,
        best_proba
    )


def create_business_summary(latest_recommendation_table):
    summary = (
        latest_recommendation_table
        .groupby("category")
        .agg(
            n_products=("product_id", "count"),
            predicted_high_products=("predicted_high_performance", "sum"),
            actual_high_products=("high_future_performance", "sum"),
            avg_probability=("probability_high_performance", "mean"),
            avg_future_revenue=("future_3m_revenue", "mean")
        )
        .reset_index()
        .sort_values("predicted_high_products", ascending=False)
    )

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "final_latest_category_recommendation_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return summary


def create_model_detail_table(classification_results):
    rows = []

    for _, row in classification_results.iterrows():
        rows.append({
            "task": "Binary classification",
            "target": "high_future_performance",
            "model": row["model"],
            "stopping_condition": "Dừng theo điều kiện hội tụ nội bộ của mô hình hoặc khi đạt số cây, số vòng lặp, số lần lặp tối đa đã thiết lập.",
            "hyperparameter_optimization": (
                "Sử dụng RandomizedSearchCV với cross-validation theo thời gian trên tập train. "
                "Ngưỡng phân lớp được chọn trên validation fold cuối của tập train theo F1-score."
            ),
            "selected_threshold": row["threshold"],
            "best_params": row["best_params"],
            "main_metrics": (
                f"Accuracy={row['Accuracy']:.4f}, "
                f"Balanced Accuracy={row['Balanced_Accuracy']:.4f}, "
                f"Precision={row['Precision']:.4f}, "
                f"Recall={row['Recall']:.4f}, "
                f"Specificity={row['Specificity']:.4f}, "
                f"F1={row['F1']:.4f}, "
                f"ROC-AUC={row['ROC_AUC']:.4f}, "
                f"PR-AUC={row['PR_AUC']:.4f}, "
                f"Brier={row['Brier_Score']:.4f}"
            ),
            "ranking_metrics": (
                f"Precision@Top10%={row['Precision_at_Top10pct']:.4f}, "
                f"Recall@Top10%={row['Recall_at_Top10pct']:.4f}, "
                f"Lift@Top10%={row['Lift_at_Top10pct']:.4f}, "
                f"Precision@Top25%={row['Precision_at_Top25pct']:.4f}, "
                f"Recall@Top25%={row['Recall_at_Top25pct']:.4f}, "
                f"Lift@Top25%={row['Lift_at_Top25pct']:.4f}"
            ),
            "confusion_matrix": (
                f"TN={int(row['TN'])}, FP={int(row['FP'])}, "
                f"FN={int(row['FN'])}, TP={int(row['TP'])}"
            ),
            "speed": (
                f"train_time_seconds={row['train_time_seconds']:.4f}, "
                f"predict_time_ms={row['predict_time_ms']:.4f}"
            ),
            "business_role": (
                "Dự đoán sản phẩm có khả năng thuộc nhóm hiệu quả kinh doanh cao trong 3 tháng tiếp theo "
                "để hỗ trợ ưu tiên danh mục sản phẩm."
            )
        })

    detail = pd.DataFrame(rows)

    detail.to_csv(
        os.path.join(OUTPUT_DIR, "final_model_detail_checklist.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return detail


# ============================================================
# 15. Main
# ============================================================

def main():
    total_start = time.time()

    orders, order_items, products = load_data()

    data_quality = create_data_quality_summary(orders, order_items, products)

    complete_months, month_coverage = get_complete_months(orders)

    monthly_sales = build_monthly_sales(
        orders,
        order_items,
        products,
        complete_months
    )

    supervised, label_distribution = create_rolling_supervised_dataset(
        monthly_sales,
        products,
        min_history_months=6,
        horizon_months=3,
        positive_fraction=0.25
    )

    train_df, test_df, train_folds, test_folds = temporal_train_test_split(
        supervised,
        n_test_folds=3
    )

    numeric_features, categorical_features = get_features()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train = train_df[numeric_features + categorical_features]
    X_test = test_df[numeric_features + categorical_features]

    y_train = train_df["high_future_performance"]
    y_test = test_df["high_future_performance"]

    print("\nDataset supervised:")
    print(f"Total rows: {len(supervised)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train folds: {train_folds}")
    print(f"Test folds: {test_folds}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    plot_label_distribution(label_distribution)

    cv_splits = create_temporal_cv_splits(train_df, n_cv_folds=3)

    model_specs = get_classification_models(preprocessor)

    SKIP_MODELS = [
        "TabM 2025 Experimental",
    ]

    for skipped_model in SKIP_MODELS:
        if skipped_model in model_specs:
            print(f"Skip slow/unstable model: {skipped_model}")
            model_specs.pop(skipped_model)

    class_results = []
    all_predictions = []
    fitted_models = {}

    best_model = None
    best_model_name = None
    best_threshold = 0.5
    best_f1 = -1

    for model_name, spec in model_specs.items():
        print(f"Training classifier with tuning: {model_name}")

        try:
            fitted_info = fit_model_with_tuning(
                model_name=model_name,
                estimator=spec["estimator"],
                param_grid=spec["params"],
                X_train=X_train,
                y_train=y_train,
                train_df=train_df,
                cv_splits=cv_splits
            )

            result, pred_df = evaluate_classifier(
                model_name=model_name,
                fitted_model=fitted_info["model"],
                threshold=fitted_info["threshold"],
                best_params=fitted_info["best_params"],
                best_cv_f1=fitted_info["best_cv_f1"],
                validation_threshold_f1=fitted_info["validation_threshold_f1"],
                validation_threshold_precision=fitted_info["validation_threshold_precision"],
                validation_threshold_recall=fitted_info["validation_threshold_recall"],
                train_time_seconds=fitted_info["train_time_seconds"],
                X_test=X_test,
                y_test=y_test
            )

            class_results.append(result)
            all_predictions.append(pred_df)
            fitted_models[model_name] = fitted_info

            if result["F1"] > best_f1:
                best_f1 = result["F1"]
                best_model = fitted_info["model"]
                best_model_name = model_name
                best_threshold = fitted_info["threshold"]

        except Exception as e:
            print(f"Bỏ qua {model_name} vì lỗi: {e}")
            SKIPPED_MODELS.append({"model": model_name, "reason": str(e)})

    if len(class_results) == 0:
        raise RuntimeError("Không có mô hình nào chạy thành công. Kiểm tra package và dữ liệu đầu vào.")

    class_results_df = pd.DataFrame(class_results).sort_values(
        "F1",
        ascending=False
    )

    class_predictions_df = pd.concat(all_predictions, ignore_index=True)

    class_results_df.to_csv(
        os.path.join(OUTPUT_DIR, "final_classification_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    class_predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "final_classification_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(SKIPPED_MODELS).to_csv(
        os.path.join(OUTPUT_DIR, "benchmark08_skipped_models.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plot_classification_results(class_results_df)

    (
        recommendation_table,
        latest_recommendation_table,
        false_positive,
        false_negative,
        true_positive,
        true_negative,
        best_proba
    ) = create_recommendation_and_error_tables(
        test_df=test_df,
        X_test=X_test,
        y_test=y_test,
        best_model=best_model,
        best_model_name=best_model_name,
        best_threshold=best_threshold
    )

    plot_best_model_curves(y_test, best_proba, best_model_name)

    feature_importance = save_feature_importance(
        best_model,
        best_model_name
    )

    category_summary = create_business_summary(latest_recommendation_table)

    model_detail = create_model_detail_table(class_results_df)

    total_time = time.time() - total_start

    print("\n" + "=" * 90)
    print("HOÀN THÀNH BENCHMARK MỞ RỘNG 08 - PHÂN LỚP SẢN PHẨM HIỆU QUẢ CAO")
    print("=" * 90)

    print("\nTóm tắt chất lượng dữ liệu:")
    print(data_quality.to_string(index=False))

    print("\nKết quả phân lớp:")
    print(class_results_df.round(4).to_string(index=False))

    print("\nMô hình tốt nhất theo F1:")
    print(best_model_name)
    print(f"Ngưỡng phân lớp đã chọn: {best_threshold:.4f}")

    print("\nTop 20 sản phẩm được mô hình dự đoán có xác suất hiệu quả cao ở cutoff mới nhất:")
    print(
        latest_recommendation_table[[
            "product_id",
            "product_name",
            "category",
            "cutoff_month",
            "target_start_month",
            "target_end_month",
            "probability_high_performance",
            "predicted_high_performance",
            "high_future_performance",
            "future_3m_revenue",
            "future_3m_quantity"
        ]]
        .head(20)
        .round(4)
        .to_string(index=False)
    )

    print("\nSố lượng false positive:", len(false_positive))
    print("Số lượng false negative:", len(false_negative))
    print("Số lượng true positive:", len(true_positive))
    print("Số lượng true negative:", len(true_negative))

    print("\nTóm tắt khuyến nghị theo category ở cutoff mới nhất:")
    print(category_summary.round(4).to_string(index=False))

    print(f"\nTổng thời gian chạy: {total_time:.2f} giây")

    print("\nCác file kết quả chính:")
    print("outputs_benchmark08/final_classification_results.csv")
    print("outputs_benchmark08/final_model_detail_checklist.csv")
    print("outputs_benchmark08/final_latest_recommendation_table.csv")
    print("outputs_benchmark08/final_false_positive_cases.csv")
    print("outputs_benchmark08/final_false_negative_cases.csv")
    print("outputs_benchmark08/final_best_model_feature_importance.csv")
    print("outputs_benchmark08/final_month_coverage_check.csv")
    print("outputs_benchmark08/final_label_distribution_by_fold.csv")

    print("\nCác hình chính:")
    print("figures_benchmark08/final_classification_f1_comparison.png")
    print("figures_benchmark08/final_classification_roc_auc_comparison.png")
    print("figures_benchmark08/final_best_confusion_matrix.png")
    print("figures_benchmark08/final_best_model_roc_curve.png")
    print("figures_benchmark08/final_best_model_pr_curve.png")
    print("figures_benchmark08/final_best_model_feature_importance.png")
    print("figures_benchmark08/final_label_distribution_by_fold.png")


if __name__ == "__main__":
    main()