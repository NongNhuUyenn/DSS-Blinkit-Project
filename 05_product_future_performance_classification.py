import os
import time
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 1. Tìm file
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
        order_items[col] = pd.to_numeric(order_items[col], errors="coerce").fillna(0)

    for col in ["price", "mrp", "margin_percentage", "shelf_life_days"]:
        if col in products.columns:
            products[col] = pd.to_numeric(products[col], errors="coerce").fillna(0)

    print("Đã đọc dữ liệu:")
    print(f"Orders clean: {orders.shape} - {orders_path}")
    print(f"Order items:  {order_items.shape} - {order_items_path}")
    print(f"Products:     {products.shape} - {products_path}")

    return orders, order_items, products


# ============================================================
# 3. Tạo sales monthly panel
# ============================================================

def build_monthly_sales(orders, order_items, products):
    order_time = orders[["order_id", "order_date"]].copy()

    sales = order_items.merge(order_time, on="order_id", how="left")
    sales = sales.dropna(subset=["order_date"])

    sales["sales_amount"] = sales["quantity"] * sales["unit_price"]
    sales["order_month"] = sales["order_date"].dt.to_period("M")

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
    all_months = pd.period_range(
        monthly_sales["order_month"].min(),
        monthly_sales["order_month"].max(),
        freq="M"
    )

    full_index = pd.MultiIndex.from_product(
        [product_ids, all_months],
        names=["product_id", "order_month"]
    )

    monthly_sales = (
        monthly_sales
        .set_index(["product_id", "order_month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return monthly_sales, all_months


# ============================================================
# 4. Tạo supervised dataset theo rolling cutoff
# ============================================================

def safe_divide(a, b):
    return np.where(b != 0, a / b, 0)


def create_rolling_supervised_dataset(monthly_sales, products, min_history_months=6, horizon_months=3):
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
                fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce").fillna(0)

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

        threshold = fold_df["future_3m_revenue"].quantile(0.75)

        fold_df["high_future_performance"] = (
            fold_df["future_3m_revenue"] >= threshold
        ).astype(int)

        fold_df["cutoff_month"] = cutoff_month
        fold_df["target_start_month"] = target_start_month
        fold_df["target_end_month"] = target_end_month
        fold_df["fold_id"] = cutoff_idx

        rows.append(fold_df)

    supervised = pd.concat(rows, ignore_index=True)

    supervised = supervised.replace([np.inf, -np.inf], np.nan).fillna(0)

    supervised.to_csv(
        os.path.join(OUTPUT_DIR, "product_future_performance_supervised_dataset.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return supervised


# ============================================================
# 5. Train/test split theo thời gian
# ============================================================

def temporal_train_test_split(df, n_test_folds=3):
    fold_ids = sorted(df["fold_id"].unique())

    test_folds = fold_ids[-n_test_folds:]
    train_folds = fold_ids[:-n_test_folds]

    train_df = df[df["fold_id"].isin(train_folds)].copy()
    test_df = df[df["fold_id"].isin(test_folds)].copy()

    return train_df, test_df


# ============================================================
# 6. Classification models
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
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )


def get_classification_models(preprocessor):
    models = {

        "Logistic Regression Balanced": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Linear SVM Balanced": Pipeline([
            ("preprocess", preprocessor),
            ("model", SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Decision Tree Balanced": Pipeline([
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest Balanced": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),

        "Extra Trees Balanced": Pipeline([
            ("preprocess", preprocessor),
            ("model", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("preprocess", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ]),

        "HistGradientBoosting": Pipeline([
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


# ============================================================
# 7. Classification evaluation
# ============================================================

def get_positive_probability(model, X):
    if hasattr(model.named_steps["model"], "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model.named_steps["model"], "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    return model.predict(X)


def evaluate_classifier(model_name, model, X_train, y_train, X_test, y_test):
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    y_proba = get_positive_probability(model, X_test)
    predict_time = time.time() - start_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    k = int(y_test.sum())
    precision_at_k, recall_at_k, lift_at_k = ranking_metrics_at_k(
        y_test,
        y_proba,
        k
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    result = {
        "model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "PR_AUC": average_precision_score(y_test, y_proba),
        "Brier_Score": brier_score_loss(y_test, y_proba),
        "Precision_at_K": precision_at_k,
        "Recall_at_K": recall_at_k,
        "Lift_at_K": lift_at_k,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "train_time_seconds": train_time,
        "predict_time_seconds": predict_time
    }

    pred_df = pd.DataFrame({
        "model": model_name,
        "actual": y_test.values,
        "predicted": y_pred,
        "probability_high_performance": y_proba
    })

    return result, pred_df, model





# ============================================================
# 9. Vẽ hình
# ============================================================

def plot_classification_results(results_df):
    plot_df = results_df.sort_values("F1", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["model"], plot_df["F1"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("F1-score")
    plt.title("So sánh mô hình phân lớp theo F1-score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "future_performance_classification_f1_comparison.png"),
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
        os.path.join(FIGURE_DIR, "future_performance_best_confusion_matrix.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 10. Model detail theo checklist
# ============================================================

def create_model_detail_table(classification_results):
    rows = []

    for _, row in classification_results.iterrows():
        rows.append({
            "task": "Binary classification",
            "target": "high_future_performance",
            "model": row["model"],
            "stopping_condition": "Dừng theo điều kiện hội tụ nội bộ của mô hình hoặc khi đạt số cây, số vòng lặp, hoặc số lần lặp tối đa đã thiết lập.",
            "hyperparameter_optimization": "Phiên bản hiện tại so sánh nhiều kiến trúc mô hình với cấu hình cố định. Các mô hình Logistic Regression, SVM, Decision Tree, Random Forest và Extra Trees sử dụng class_weight='balanced' để xử lý lệch lớp.",
            "main_metrics": f"Accuracy={row['Accuracy']:.4f}, Balanced Accuracy={row['Balanced_Accuracy']:.4f}, Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, Specificity={row['Specificity']:.4f}, F1={row['F1']:.4f}, ROC-AUC={row['ROC_AUC']:.4f}, PR-AUC={row['PR_AUC']:.4f}, Brier={row['Brier_Score']:.4f}",
            "confusion_matrix": f"TN={int(row['TN'])}, FP={int(row['FP'])}, FN={int(row['FN'])}, TP={int(row['TP'])}",
            "business_role": "Dự đoán sản phẩm có khả năng thuộc nhóm hiệu quả kinh doanh cao trong 3 tháng tiếp theo để hỗ trợ ưu tiên danh mục sản phẩm."
        })

    detail = pd.DataFrame(rows)

    detail.to_csv(
        os.path.join(OUTPUT_DIR, "future_performance_model_detail_checklist.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return detail







def ranking_metrics_at_k(y_true, y_proba, k):
    ranking_df = pd.DataFrame({
        "actual": y_true.values,
        "probability": y_proba
    }).sort_values("probability", ascending=False)

    top_k = ranking_df.head(k)

    precision_at_k = top_k["actual"].mean()

    total_positive = ranking_df["actual"].sum()
    recall_at_k = top_k["actual"].sum() / total_positive if total_positive > 0 else 0

    base_rate = ranking_df["actual"].mean()
    lift_at_k = precision_at_k / base_rate if base_rate > 0 else 0

    return precision_at_k, recall_at_k, lift_at_k

# ============================================================
# 11. Main
# ============================================================



def main():
    orders, order_items, products = load_data()

    monthly_sales, all_months = build_monthly_sales(orders, order_items, products)

    supervised = create_rolling_supervised_dataset(
        monthly_sales,
        products,
        min_history_months=6,
        horizon_months=3
    )

    train_df, test_df = temporal_train_test_split(supervised, n_test_folds=3)

    numeric_features, categorical_features = get_features()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train = train_df[numeric_features + categorical_features]
    X_test = test_df[numeric_features + categorical_features]

    y_train_cls = train_df["high_future_performance"]
    y_test_cls = test_df["high_future_performance"]

    print("\nDataset supervised:")
    print(f"Total rows: {len(supervised)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train positive rate: {y_train_cls.mean():.4f}")
    print(f"Test positive rate: {y_test_cls.mean():.4f}")

    # Classification
    class_models = get_classification_models(preprocessor)

    class_results = []
    all_predictions = []

    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, model in class_models.items():
        print(f"Training classifier: {model_name}")

        result, pred_df, fitted_model = evaluate_classifier(
            model_name,
            model,
            X_train,
            y_train_cls,
            X_test,
            y_test_cls
        )

        class_results.append(result)
        all_predictions.append(pred_df)

        if result["F1"] > best_f1:
            best_f1 = result["F1"]
            best_model = fitted_model
            best_model_name = model_name

    class_results_df = pd.DataFrame(class_results).sort_values(
        "F1",
        ascending=False
    )

    class_predictions_df = pd.concat(all_predictions, ignore_index=True)

    class_results_df.to_csv(
        os.path.join(OUTPUT_DIR, "future_performance_classification_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    class_predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "future_performance_classification_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plot_classification_results(class_results_df)

    # ============================================================
    # Recommendation table using best model
    # ============================================================

    best_proba = get_positive_probability(best_model, X_test)
    best_pred = best_model.predict(X_test)

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
        "high_future_performance"
    ]].copy()

    # Phải thêm 2 cột này trước khi phân tích lỗi
    recommendation_table["predicted_high_performance"] = best_pred
    recommendation_table["probability_high_performance"] = best_proba

    recommendation_table = recommendation_table.sort_values(
        "probability_high_performance",
        ascending=False
    ).reset_index(drop=True)

    recommendation_table.to_csv(
        os.path.join(OUTPUT_DIR, "future_performance_recommendation_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # Latest cutoff recommendation table
    # ============================================================

    latest_cutoff = recommendation_table["cutoff_month"].max()

    latest_recommendation_table = (
        recommendation_table[
            recommendation_table["cutoff_month"] == latest_cutoff
        ]
        .sort_values("probability_high_performance", ascending=False)
        .reset_index(drop=True)
    )

    latest_recommendation_table.to_csv(
        os.path.join(OUTPUT_DIR, "latest_future_performance_recommendation_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # Error analysis
    # ============================================================

    error_analysis = recommendation_table.copy()

    false_positive = error_analysis[
        (error_analysis["predicted_high_performance"] == 1)
        & (error_analysis["high_future_performance"] == 0)
    ].sort_values("probability_high_performance", ascending=False)

    false_negative = error_analysis[
        (error_analysis["predicted_high_performance"] == 0)
        & (error_analysis["high_future_performance"] == 1)
    ].sort_values("future_3m_revenue", ascending=False)

    false_positive.to_csv(
        os.path.join(OUTPUT_DIR, "future_performance_false_positive_cases.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    false_negative.to_csv(
        os.path.join(OUTPUT_DIR, "future_performance_false_negative_cases.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # Model detail checklist
    # ============================================================

    detail = create_model_detail_table(class_results_df)

    # ============================================================
    # Print results
    # ============================================================

    print("\n" + "=" * 90)
    print("HOÀN THÀNH BÀI TOÁN PHÂN LỚP SẢN PHẨM HIỆU QUẢ CAO")
    print("=" * 90)

    print("\nKết quả phân lớp:")
    print(class_results_df.round(4).to_string(index=False))

    print("\nMô hình phân lớp tốt nhất theo F1:")
    print(best_model_name)

    print("\nTop 15 sản phẩm được mô hình dự đoán có xác suất hiệu quả cao ở cutoff mới nhất:")

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
            "future_3m_revenue"
        ]]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    print("\nSố lượng false positive:", len(false_positive))
    print("Số lượng false negative:", len(false_negative))

    print("\nFile kết quả đã lưu trong thư mục outputs.")
    print("Hình đã lưu trong thư mục figures.")


if __name__ == "__main__":
    main()