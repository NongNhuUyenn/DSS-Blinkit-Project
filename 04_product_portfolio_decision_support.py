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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42


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

    raise FileNotFoundError(
        f"Không tìm thấy file {filename}. "
        f"Hãy đặt file trong thư mục project hoặc thư mục blinkit_data2."
    )


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

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        errors="coerce"
    )

    for col in ["quantity", "unit_price"]:
        order_items[col] = pd.to_numeric(
            order_items[col],
            errors="coerce"
        ).fillna(0)

    product_numeric_cols = [
        "price",
        "mrp",
        "margin_percentage",
        "shelf_life_days",
        "min_stock_level",
        "max_stock_level"
    ]

    for col in product_numeric_cols:
        if col in products.columns:
            products[col] = pd.to_numeric(
                products[col],
                errors="coerce"
            ).fillna(0)

    print("Đã đọc dữ liệu:")
    print(f"Orders clean: {orders.shape} - {orders_path}")
    print(f"Order items:  {order_items.shape} - {order_items_path}")
    print(f"Products:     {products.shape} - {products_path}")

    print("\nOK: Code này chỉ dùng orders_clean, order_items và products.")
    print("OK: Không dùng inventory, feedback, delivery hoặc marketing.")

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
        os.path.join(OUTPUT_DIR, "portfolio_data_quality_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# ============================================================
# 4. Hàm chuẩn hóa
# ============================================================

def minmax(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    s_min = s.min()
    s_max = s.max()

    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - s_min) / (s_max - s_min)


def safe_divide(a, b):
    return np.where(b != 0, a / b, 0)


# ============================================================
# 5. Xây dựng bảng cấp sản phẩm
# ============================================================

def build_product_portfolio_dataset(orders, order_items, products):
    order_time = orders[["order_id", "order_date"]].copy()

    sales = order_items.merge(
        order_time,
        on="order_id",
        how="left"
    )

    sales = sales.dropna(subset=["order_date"])

    sales["sales_amount"] = sales["quantity"] * sales["unit_price"]
    sales["order_month"] = sales["order_date"].dt.to_period("M")

    # ------------------------------------------------------------
    # 5.1. Tổng hợp bán hàng theo sản phẩm
    # ------------------------------------------------------------

    sales_agg = (
        sales
        .groupby("product_id")
        .agg(
            total_quantity_sold=("quantity", "sum"),
            total_revenue=("sales_amount", "sum"),
            total_order_lines=("order_id", "count"),
            total_orders=("order_id", "nunique"),
            average_selling_price=("unit_price", "mean"),
            first_sale_date=("order_date", "min"),
            last_sale_date=("order_date", "max")
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # 5.2. Tạo panel nhu cầu theo tháng
    # ------------------------------------------------------------

    all_products = products["product_id"].drop_duplicates()

    if len(sales) > 0:
        all_months = pd.period_range(
            sales["order_month"].min(),
            sales["order_month"].max(),
            freq="M"
        )
    else:
        all_months = pd.period_range("2024-01", "2024-01", freq="M")

    full_index = pd.MultiIndex.from_product(
        [all_products, all_months],
        names=["product_id", "order_month"]
    )

    monthly_demand = (
        sales
        .groupby(["product_id", "order_month"])
        .agg(
            monthly_quantity_sold=("quantity", "sum"),
            monthly_revenue=("sales_amount", "sum")
        )
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    demand_stats = (
        monthly_demand
        .groupby("product_id")
        .agg(
            mean_monthly_demand=("monthly_quantity_sold", "mean"),
            std_monthly_demand=("monthly_quantity_sold", "std"),
            max_monthly_demand=("monthly_quantity_sold", "max"),
            active_months=("monthly_quantity_sold", lambda x: int((x > 0).sum())),
            mean_monthly_revenue=("monthly_revenue", "mean"),
            std_monthly_revenue=("monthly_revenue", "std")
        )
        .reset_index()
    )

    demand_stats["std_monthly_demand"] = demand_stats["std_monthly_demand"].fillna(0)
    demand_stats["std_monthly_revenue"] = demand_stats["std_monthly_revenue"].fillna(0)

    demand_stats["demand_cv"] = safe_divide(
        demand_stats["std_monthly_demand"],
        demand_stats["mean_monthly_demand"]
    )

    demand_stats["revenue_cv"] = safe_divide(
        demand_stats["std_monthly_revenue"],
        demand_stats["mean_monthly_revenue"]
    )

    # Nếu sản phẩm không bán được tháng nào, coi là biến động cao
    max_cv = demand_stats.loc[
        np.isfinite(demand_stats["demand_cv"]),
        "demand_cv"
    ].max()

    if pd.isna(max_cv):
        max_cv = 0

    demand_stats["demand_cv"] = (
        demand_stats["demand_cv"]
        .replace([np.inf, -np.inf], max_cv)
        .fillna(max_cv)
    )

    demand_stats["revenue_cv"] = (
        demand_stats["revenue_cv"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # ------------------------------------------------------------
    # 5.3. Gộp với bảng sản phẩm
    # ------------------------------------------------------------

    df = products.copy()

    df = df.merge(sales_agg, on="product_id", how="left")
    df = df.merge(demand_stats, on="product_id", how="left")

    numeric_cols = [
        "total_quantity_sold",
        "total_revenue",
        "total_order_lines",
        "total_orders",
        "average_selling_price",
        "mean_monthly_demand",
        "std_monthly_demand",
        "max_monthly_demand",
        "active_months",
        "mean_monthly_revenue",
        "std_monthly_revenue",
        "demand_cv",
        "revenue_cv"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["estimated_profit"] = (
        df["total_revenue"] * df["margin_percentage"] / 100
    )

    df["discount_rate"] = safe_divide(
        df["mrp"] - df["price"],
        df["mrp"]
    )

    df["discount_rate"] = np.clip(df["discount_rate"], 0, 1)

    df["price_gap"] = df["mrp"] - df["price"]

    df["days_between_first_last_sale"] = (
        pd.to_datetime(df["last_sale_date"], errors="coerce")
        - pd.to_datetime(df["first_sale_date"], errors="coerce")
    ).dt.days.fillna(0)

    df["sales_frequency_score"] = safe_divide(
        df["active_months"],
        len(all_months)
    )

    monthly_demand["order_month"] = monthly_demand["order_month"].astype(str)

    monthly_demand.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_monthly_product_demand.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return df, monthly_demand


# ============================================================
# 6. ABC Analysis
# ============================================================

def add_abc_class(df):
    result = df.copy()

    result = result.sort_values(
        "total_revenue",
        ascending=False
    ).reset_index(drop=True)

    total_revenue = result["total_revenue"].sum()

    if total_revenue <= 0:
        result["revenue_share"] = 0
        result["cumulative_revenue_share"] = 0
        result["abc_class"] = "C"
        return result

    result["revenue_share"] = result["total_revenue"] / total_revenue
    result["cumulative_revenue_share"] = result["revenue_share"].cumsum()

    result["abc_class"] = np.select(
        [
            result["cumulative_revenue_share"] <= 0.80,
            result["cumulative_revenue_share"] <= 0.95
        ],
        ["A", "B"],
        default="C"
    )

    return result


# ============================================================
# 7. XYZ Analysis
# ============================================================

def add_xyz_class(df):
    result = df.copy()

    # CV càng thấp, nhu cầu càng ổn định.
    cv_rank = result["demand_cv"].rank(
        method="average",
        pct=True,
        ascending=True
    )

    result["xyz_class"] = np.select(
        [
            cv_rank <= 1 / 3,
            cv_rank <= 2 / 3
        ],
        ["X", "Y"],
        default="Z"
    )

    return result


# ============================================================
# 8. Điểm ưu tiên danh mục sản phẩm
# ============================================================

def add_priority_score(df):
    result = df.copy()

    result["revenue_score"] = minmax(np.log1p(result["total_revenue"]))
    result["quantity_score"] = minmax(np.log1p(result["total_quantity_sold"]))
    result["profit_score"] = minmax(np.log1p(result["estimated_profit"].clip(lower=0)))
    result["margin_score"] = minmax(result["margin_percentage"])

    # Nhu cầu càng ổn định thì điểm càng cao
    result["stability_score"] = 1 - minmax(result["demand_cv"])

    # Hạn sử dụng ngắn không phải tồn kho thực tế, nhưng là yếu tố cần chú ý trong danh mục
    result["shelf_life_attention_score"] = 1 - minmax(result["shelf_life_days"])

    result["discount_score"] = minmax(result["discount_rate"])

    result["priority_score"] = (
        0.30 * result["revenue_score"]
        + 0.25 * result["profit_score"]
        + 0.20 * result["quantity_score"]
        + 0.10 * result["stability_score"]
        + 0.10 * result["shelf_life_attention_score"]
        + 0.05 * result["margin_score"]
    )

    score_rank = result["priority_score"].rank(
        method="average",
        pct=True,
        ascending=True
    )

    result["priority_level"] = np.select(
        [
            score_rank > 0.75,
            score_rank > 0.50,
            score_rank > 0.25
        ],
        ["Critical", "High", "Medium"],
        default="Low"
    )

    return result


# ============================================================
# 9. Khuyến nghị hành động
# ============================================================

def recommend_action(row):
    abc = row["abc_class"]
    xyz = row["xyz_class"]

    shelf_life_short = row["shelf_life_days"] <= row["_shelf_life_q25"]
    high_margin = row["margin_percentage"] >= row["_margin_q75"]
    high_discount = row["discount_rate"] >= row["_discount_q75"]
    low_revenue = row["total_revenue"] <= row["_revenue_q25"]

    if abc == "A" and xyz == "X":
        return "Core product: ưu tiên duy trì và đảm bảo luôn có trong danh mục."

    if abc == "A" and xyz in ["Y", "Z"]:
        return "High-value volatile product: duy trì nhưng cần theo dõi biến động nhu cầu."

    if abc == "B" and high_margin:
        return "Growth candidate: sản phẩm có lợi nhuận tốt, có thể cân nhắc thúc đẩy bán hàng."

    if abc == "C" and xyz == "Z":
        return "Review candidate: đóng góp thấp và nhu cầu biến động cao, cần xem xét lại mức ưu tiên."

    if shelf_life_short and abc in ["A", "B"]:
        return "Shelf-life attention: sản phẩm có giá trị nhưng hạn sử dụng ngắn, cần kiểm soát vòng đời bán."

    if high_discount and low_revenue:
        return "Pricing review: giảm giá cao nhưng doanh thu thấp, cần xem xét lại chính sách giá."

    if abc == "C":
        return "Low contribution: duy trì ở mức ưu tiên thấp hoặc đánh giá lại danh mục."

    return "Standard management: duy trì chính sách danh mục thông thường."


def add_recommendations(df):
    result = df.copy()

    result["_shelf_life_q25"] = result["shelf_life_days"].quantile(0.25)
    result["_margin_q75"] = result["margin_percentage"].quantile(0.75)
    result["_discount_q75"] = result["discount_rate"].quantile(0.75)
    result["_revenue_q25"] = result["total_revenue"].quantile(0.25)

    result["recommended_action"] = result.apply(recommend_action, axis=1)

    result = result.drop(
        columns=[
            "_shelf_life_q25",
            "_margin_q75",
            "_discount_q75",
            "_revenue_q25"
        ]
    )

    return result


# ============================================================
# 10. KMeans phân cụm sản phẩm
# ============================================================

def add_kmeans_clusters(df):
    result = df.copy()

    feature_df = pd.DataFrame({
        "log_total_revenue": np.log1p(result["total_revenue"]),
        "log_total_quantity_sold": np.log1p(result["total_quantity_sold"]),
        "log_estimated_profit": np.log1p(result["estimated_profit"].clip(lower=0)),
        "demand_cv": result["demand_cv"],
        "discount_rate": result["discount_rate"],
        "shelf_life_days": result["shelf_life_days"],
        "margin_percentage": result["margin_percentage"],
        "priority_score": result["priority_score"]
    })

    feature_df = (
        feature_df
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

    k_candidates = [3, 4, 5, 6]
    silhouette_rows = []

    best_k = 3
    best_score = -1

    for k in k_candidates:
        if len(result) <= k:
            continue

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
            max_iter=300,
            algorithm="lloyd"
        )

        labels = model.fit_predict(X)

        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1

        silhouette_rows.append({
            "k": k,
            "silhouette_score": score
        })

        if score > best_score:
            best_score = score
            best_k = k

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
        algorithm="lloyd"
    )

    result["kmeans_cluster"] = final_model.fit_predict(X)

    silhouette_df = pd.DataFrame(silhouette_rows)

    cluster_profile = (
        result
        .groupby("kmeans_cluster")
        .agg(
            n_products=("product_id", "count"),
            avg_priority_score=("priority_score", "mean"),
            avg_revenue=("total_revenue", "mean"),
            avg_quantity_sold=("total_quantity_sold", "mean"),
            avg_profit=("estimated_profit", "mean"),
            avg_demand_cv=("demand_cv", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
            avg_margin=("margin_percentage", "mean"),
            avg_shelf_life_days=("shelf_life_days", "mean")
        )
        .reset_index()
        .sort_values("avg_priority_score", ascending=False)
    )

    cluster_profile["cluster_priority_rank"] = np.arange(
        1,
        len(cluster_profile) + 1
    )

    result = result.merge(
        cluster_profile[["kmeans_cluster", "cluster_priority_rank"]],
        on="kmeans_cluster",
        how="left"
    )

    silhouette_df.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_kmeans_silhouette_scores.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    cluster_profile.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_kmeans_cluster_profile.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return result, silhouette_df, cluster_profile, best_k, best_score


# ============================================================
# 11. Tạo bảng kết quả
# ============================================================

def create_summary_tables(df):
    decision_cols = [
        "product_id",
        "product_name",
        "category",
        "brand",
        "price",
        "mrp",
        "margin_percentage",
        "shelf_life_days",
        "total_quantity_sold",
        "total_revenue",
        "estimated_profit",
        "average_selling_price",
        "discount_rate",
        "mean_monthly_demand",
        "std_monthly_demand",
        "demand_cv",
        "active_months",
        "sales_frequency_score",
        "abc_class",
        "xyz_class",
        "priority_score",
        "priority_level",
        "kmeans_cluster",
        "cluster_priority_rank",
        "recommended_action"
    ]

    decision_table = df[decision_cols].sort_values(
        ["priority_score", "total_revenue"],
        ascending=[False, False]
    )

    decision_table.to_csv(
        os.path.join(OUTPUT_DIR, "product_portfolio_decision_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    top_priority = decision_table.head(30)

    top_priority.to_csv(
        os.path.join(OUTPUT_DIR, "top_priority_products.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    abc_xyz_summary = (
        df
        .groupby(["abc_class", "xyz_class"])
        .agg(
            n_products=("product_id", "count"),
            total_revenue=("total_revenue", "sum"),
            total_quantity_sold=("total_quantity_sold", "sum"),
            avg_profit=("estimated_profit", "mean"),
            avg_demand_cv=("demand_cv", "mean"),
            avg_priority_score=("priority_score", "mean")
        )
        .reset_index()
        .sort_values(["abc_class", "xyz_class"])
    )

    abc_xyz_summary.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_abc_xyz_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    category_summary = (
        df
        .groupby("category")
        .agg(
            n_products=("product_id", "count"),
            total_revenue=("total_revenue", "sum"),
            total_quantity_sold=("total_quantity_sold", "sum"),
            total_profit=("estimated_profit", "sum"),
            avg_margin=("margin_percentage", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
            avg_demand_cv=("demand_cv", "mean"),
            avg_priority_score=("priority_score", "mean"),
            critical_products=("priority_level", lambda x: int((x == "Critical").sum()))
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    category_summary.to_csv(
        os.path.join(OUTPUT_DIR, "category_portfolio_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    priority_summary = (
        df
        .groupby("priority_level")
        .agg(
            n_products=("product_id", "count"),
            avg_revenue=("total_revenue", "mean"),
            avg_quantity_sold=("total_quantity_sold", "mean"),
            avg_profit=("estimated_profit", "mean"),
            avg_priority_score=("priority_score", "mean")
        )
        .reset_index()
    )

    priority_summary.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_priority_level_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return decision_table, top_priority, abc_xyz_summary, category_summary, priority_summary


# ============================================================
# 12. Bảng mô tả mô hình theo checklist
# ============================================================

def create_model_detail_table(best_k, best_silhouette):
    rows = [
        {
            "component": "ABC Analysis",
            "model_type": "Rule-based product portfolio classification",
            "input_features": "total_revenue",
            "output": "abc_class",
            "stopping_condition": "Không có huấn luyện lặp. Dừng sau khi tính doanh thu tích lũy của toàn bộ sản phẩm.",
            "hyperparameter_optimization": "Không tối ưu siêu tham số. Dùng ngưỡng ABC phổ biến theo doanh thu tích lũy.",
            "main_parameters": "A: cumulative revenue <= 80%; B: <= 95%; C: > 95%",
            "evaluation": "Kiểm tra số sản phẩm và doanh thu theo từng nhóm ABC.",
            "business_role": "Xác định nhóm sản phẩm đóng góp doanh thu chính."
        },
        {
            "component": "XYZ Analysis",
            "model_type": "Rule-based demand variability classification",
            "input_features": "mean_monthly_demand, std_monthly_demand, demand_cv",
            "output": "xyz_class",
            "stopping_condition": "Không có huấn luyện lặp. Dừng sau khi tính hệ số biến động nhu cầu theo tháng.",
            "hyperparameter_optimization": "Không tối ưu siêu tham số. Chia theo phân vị demand_cv của chính bộ dữ liệu.",
            "main_parameters": "X: CV thấp; Y: CV trung bình; Z: CV cao",
            "evaluation": "Kiểm tra phân bố sản phẩm theo nhóm X, Y, Z và ma trận ABC-XYZ.",
            "business_role": "Đánh giá độ ổn định nhu cầu để hỗ trợ quản trị danh mục."
        },
        {
            "component": "Priority Scoring",
            "model_type": "Weighted multi-criteria scoring model",
            "input_features": "revenue_score, profit_score, quantity_score, stability_score, shelf_life_attention_score, margin_score",
            "output": "priority_score, priority_level, recommended_action",
            "stopping_condition": "Không có huấn luyện lặp. Dừng sau khi tính điểm ưu tiên cho toàn bộ sản phẩm.",
            "hyperparameter_optimization": "Trọng số được thiết lập theo mục tiêu nghiệp vụ của bài toán.",
            "main_parameters": "0.30 revenue, 0.25 profit, 0.20 quantity, 0.10 stability, 0.10 shelf-life attention, 0.05 margin",
            "evaluation": "Kiểm tra top sản phẩm ưu tiên và phân bố priority_level.",
            "business_role": "Tạo điểm ưu tiên dễ diễn giải để hỗ trợ quyết định sản phẩm nào cần tập trung."
        },
        {
            "component": "KMeans Clustering",
            "model_type": "Unsupervised machine learning clustering",
            "input_features": "log revenue, log quantity, log profit, demand_cv, discount_rate, shelf_life_days, margin_percentage, priority_score",
            "output": "kmeans_cluster, cluster_priority_rank",
            "stopping_condition": "Dừng khi KMeans hội tụ hoặc đạt số vòng lặp tối đa.",
            "hyperparameter_optimization": "Thử k từ 3 đến 6 và chọn k có silhouette_score cao nhất.",
            "main_parameters": f"best_k={best_k}, best_silhouette_score={best_silhouette:.4f}, random_state=42, n_init=10",
            "evaluation": "Silhouette score và bảng cluster profile.",
            "business_role": "Phân nhóm sản phẩm có hành vi kinh doanh tương tự để hỗ trợ chính sách danh mục."
        }
    ]

    detail = pd.DataFrame(rows)

    detail.to_csv(
        os.path.join(OUTPUT_DIR, "product_portfolio_model_detail_checklist.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return detail


# ============================================================
# 13. Vẽ biểu đồ
# ============================================================

def save_bar_chart(series, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def create_figures(df, top_priority, abc_xyz_summary, category_summary, cluster_profile):
    abc_counts = df["abc_class"].value_counts().sort_index()

    save_bar_chart(
        abc_counts,
        "Phân bố sản phẩm theo nhóm ABC",
        "ABC class",
        "Số sản phẩm",
        "portfolio_abc_distribution.png"
    )

    xyz_counts = df["xyz_class"].value_counts().sort_index()

    save_bar_chart(
        xyz_counts,
        "Phân bố sản phẩm theo nhóm XYZ",
        "XYZ class",
        "Số sản phẩm",
        "portfolio_xyz_distribution.png"
    )

    priority_counts = (
        df["priority_level"]
        .value_counts()
        .reindex(["Critical", "High", "Medium", "Low"])
        .fillna(0)
    )

    save_bar_chart(
        priority_counts,
        "Phân bố sản phẩm theo mức ưu tiên",
        "Priority level",
        "Số sản phẩm",
        "portfolio_priority_level_distribution.png"
    )

    matrix = pd.crosstab(df["abc_class"], df["xyz_class"]).reindex(
        index=["A", "B", "C"],
        columns=["X", "Y", "Z"],
        fill_value=0
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(matrix.values)
    plt.xticks(np.arange(len(matrix.columns)), matrix.columns)
    plt.yticks(np.arange(len(matrix.index)), matrix.index)
    plt.xlabel("XYZ class")
    plt.ylabel("ABC class")
    plt.title("Ma trận ABC-XYZ")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix.values[i, j]), ha="center", va="center")

    plt.colorbar(label="Số sản phẩm")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "portfolio_abc_xyz_matrix.png"), dpi=300)
    plt.close()

    top_plot = top_priority.head(15).sort_values("priority_score")

    labels = (
        top_plot["product_name"].astype(str)
        + " - "
        + top_plot["product_id"].astype(str)
    )

    plt.figure(figsize=(11, 7))
    plt.barh(labels, top_plot["priority_score"])
    plt.title("Top 15 sản phẩm có điểm ưu tiên cao nhất")
    plt.xlabel("Priority score")
    plt.ylabel("Sản phẩm")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "portfolio_top_priority_products.png"), dpi=300)
    plt.close()

    category_plot = category_summary.sort_values("total_revenue", ascending=False)

    plt.figure(figsize=(11, 6))
    plt.bar(category_plot["category"], category_plot["total_revenue"])
    plt.title("Tổng doanh thu theo danh mục sản phẩm")
    plt.xlabel("Danh mục")
    plt.ylabel("Total revenue")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "portfolio_category_revenue.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["priority_score"], bins=20)
    plt.title("Phân phối điểm ưu tiên sản phẩm")
    plt.xlabel("Priority score")
    plt.ylabel("Số sản phẩm")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "portfolio_priority_score_distribution.png"), dpi=300)
    plt.close()

    cluster_plot = cluster_profile.sort_values("cluster_priority_rank")

    plt.figure(figsize=(8, 5))
    plt.bar(
        cluster_plot["kmeans_cluster"].astype(str),
        cluster_plot["avg_priority_score"]
    )
    plt.title("Điểm ưu tiên trung bình theo cụm KMeans")
    plt.xlabel("KMeans cluster")
    plt.ylabel("Average priority score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "portfolio_kmeans_cluster_profile.png"), dpi=300)
    plt.close()


# ============================================================
# 14. Main
# ============================================================

def main():
    start_time = time.time()

    orders, order_items, products = load_data()

    quality_summary = create_data_quality_summary(
        orders,
        order_items,
        products
    )

    product_df, monthly_demand = build_product_portfolio_dataset(
        orders,
        order_items,
        products
    )

    product_df = add_abc_class(product_df)
    product_df = add_xyz_class(product_df)
    product_df = add_priority_score(product_df)
    product_df = add_recommendations(product_df)

    product_df, silhouette_df, cluster_profile, best_k, best_silhouette = (
        add_kmeans_clusters(product_df)
    )

    (
        decision_table,
        top_priority,
        abc_xyz_summary,
        category_summary,
        priority_summary
    ) = create_summary_tables(product_df)

    model_detail = create_model_detail_table(best_k, best_silhouette)

    create_figures(
        product_df,
        top_priority,
        abc_xyz_summary,
        category_summary,
        cluster_profile
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 90)
    print("HOÀN THÀNH BÀI TOÁN HỖ TRỢ QUYẾT ĐỊNH DANH MỤC SẢN PHẨM")
    print("=" * 90)

    print("\nTóm tắt chất lượng dữ liệu:")
    print(quality_summary.to_string(index=False))

    print("\nPhân bố ABC:")
    print(product_df["abc_class"].value_counts().sort_index().to_string())

    print("\nPhân bố XYZ:")
    print(product_df["xyz_class"].value_counts().sort_index().to_string())

    print("\nPhân bố mức ưu tiên:")
    print(
        product_df["priority_level"]
        .value_counts()
        .reindex(["Critical", "High", "Medium", "Low"])
        .fillna(0)
        .to_string()
    )

    print("\nKMeans:")
    print(f"best_k = {best_k}")
    print(f"best_silhouette_score = {best_silhouette:.4f}")

    print("\nTop 10 sản phẩm cần ưu tiên:")

    cols_to_print = [
        "product_id",
        "product_name",
        "category",
        "total_revenue",
        "total_quantity_sold",
        "estimated_profit",
        "demand_cv",
        "abc_class",
        "xyz_class",
        "priority_score",
        "priority_level"
    ]

    print(
        top_priority[cols_to_print]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\nCác file kết quả đã lưu trong thư mục outputs.")
    print("Các hình đã lưu trong thư mục figures.")
    print(f"Thời gian chạy: {elapsed:.2f} giây")


if __name__ == "__main__":
    main()