import os

# Giảm lỗi treo hoặc chạy chậm do đa luồng trên Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. Cấu hình
# ============================================================

OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. Tìm file dữ liệu
# ============================================================

def find_file(filename):
    candidate_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join("blinkit_data2", filename),
        os.path.join("..", filename),
        os.path.join("..", "blinkit_data2", filename),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Không tìm thấy file {filename}. "
        f"Hãy đặt file trong thư mục project hoặc trong thư mục blinkit_data2."
    )


# ============================================================
# 3. Hàm chuẩn hóa min-max
# ============================================================

def minmax(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    s_min = s.min()
    s_max = s.max()

    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - s_min) / (s_max - s_min)


# ============================================================
# 4. Đọc dữ liệu
# ============================================================

def load_data():
    products_path = find_file("blinkit_products.csv")
    order_items_path = find_file("blinkit_order_items.csv")
    orders_path = find_file("blinkit_orders.csv")
    inventory_path = find_file("blinkit_inventory.csv")

    products = pd.read_csv(products_path)
    order_items = pd.read_csv(order_items_path)
    orders = pd.read_csv(orders_path)
    inventory = pd.read_csv(inventory_path)

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")

    print("Đã đọc dữ liệu:")
    print(f"Products: {products.shape} - {products_path}")
    print(f"Order items: {order_items.shape} - {order_items_path}")
    print(f"Orders: {orders.shape} - {orders_path}")
    print(f"Inventory: {inventory.shape} - {inventory_path}")

    return products, order_items, orders, inventory


# ============================================================
# 5. Kiểm tra chất lượng dữ liệu
# ============================================================

def create_data_quality_summary(products, order_items, orders, inventory):
    rows = []

    for name, df in [
        ("products", products),
        ("order_items", order_items),
        ("orders", orders),
        ("inventory", inventory),
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
        os.path.join(OUTPUT_DIR, "inventory_data_quality_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# ============================================================
# 6. Tạo bảng đặc trưng cấp sản phẩm
# ============================================================

def build_product_level_dataset(products, order_items, orders, inventory):
    # ------------------------------------------------------------
    # 6.1. Bán hàng theo sản phẩm
    # ------------------------------------------------------------

    order_with_date = orders[["order_id", "order_date"]].copy()

    sales_detail = order_items.merge(
        order_with_date,
        on="order_id",
        how="left"
    )

    sales_detail = sales_detail.dropna(subset=["order_date"])

    sales_detail["sales_amount"] = (
        sales_detail["quantity"] * sales_detail["unit_price"]
    )

    sales_detail["order_month"] = sales_detail["order_date"].dt.to_period("M")

    sales_agg = (
        sales_detail
        .groupby("product_id")
        .agg(
            total_quantity_sold=("quantity", "sum"),
            total_revenue=("sales_amount", "sum"),
            total_order_lines=("order_id", "count"),
            total_orders=("order_id", "nunique"),
            average_unit_price=("unit_price", "mean"),
            first_sale_date=("order_date", "min"),
            last_sale_date=("order_date", "max")
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # 6.2. Nhu cầu theo tháng
    # ------------------------------------------------------------

    if len(sales_detail) > 0:
        all_months = pd.period_range(
            sales_detail["order_month"].min(),
            sales_detail["order_month"].max(),
            freq="M"
        )
    else:
        all_months = pd.period_range("2024-01", "2024-01", freq="M")

    product_ids = products["product_id"].drop_duplicates()

    full_index = pd.MultiIndex.from_product(
        [product_ids, all_months],
        names=["product_id", "order_month"]
    )

    monthly_demand = (
        sales_detail
        .groupby(["product_id", "order_month"])
        .agg(monthly_quantity_sold=("quantity", "sum"))
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
            active_months=("monthly_quantity_sold", lambda x: int((x > 0).sum()))
        )
        .reset_index()
    )

    demand_stats["std_monthly_demand"] = (
        demand_stats["std_monthly_demand"].fillna(0)
    )

    demand_stats["demand_cv"] = np.where(
        demand_stats["mean_monthly_demand"] > 0,
        demand_stats["std_monthly_demand"] / demand_stats["mean_monthly_demand"],
        np.inf
    )

    demand_stats["demand_cv"] = (
        demand_stats["demand_cv"]
        .replace([np.inf, -np.inf], np.nan)
    )

    demand_stats["demand_cv"] = (
        demand_stats["demand_cv"]
        .fillna(demand_stats["demand_cv"].max())
    )

    # ------------------------------------------------------------
    # 6.3. Tồn kho theo sản phẩm
    # ------------------------------------------------------------

    inventory_agg = (
        inventory
        .groupby("product_id")
        .agg(
            total_stock_received=("stock_received", "sum"),
            total_damaged_stock=("damaged_stock", "sum"),
            inventory_records=("product_id", "count")
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # 6.4. Gộp bảng
    # ------------------------------------------------------------

    df = products.copy()

    df = df.merge(sales_agg, on="product_id", how="left")
    df = df.merge(demand_stats, on="product_id", how="left")
    df = df.merge(inventory_agg, on="product_id", how="left")

    numeric_cols = [
        "total_quantity_sold",
        "total_revenue",
        "total_order_lines",
        "total_orders",
        "average_unit_price",
        "mean_monthly_demand",
        "std_monthly_demand",
        "max_monthly_demand",
        "active_months",
        "demand_cv",
        "total_stock_received",
        "total_damaged_stock",
        "inventory_records"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in [
        "price",
        "mrp",
        "margin_percentage",
        "shelf_life_days",
        "min_stock_level",
        "max_stock_level"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["estimated_profit"] = (
        df["total_revenue"] * df["margin_percentage"] / 100
    )

    df["damage_rate"] = np.where(
        df["total_stock_received"] > 0,
        df["total_damaged_stock"] / df["total_stock_received"],
        0
    )

    df["damage_rate"] = df["damage_rate"].clip(lower=0, upper=1)

    df["estimated_net_stock"] = (
        df["total_stock_received"]
        - df["total_damaged_stock"]
        - df["total_quantity_sold"]
    )

    df["stock_shortage_amount"] = np.maximum(
        df["min_stock_level"] - df["estimated_net_stock"],
        0
    )

    df["overstock_amount"] = np.maximum(
        df["estimated_net_stock"] - df["max_stock_level"],
        0
    )

    return df, monthly_demand


# ============================================================
# 7. ABC Analysis
# ============================================================

def add_abc_class(df):
    result = df.copy()

    result = result.sort_values(
        "total_revenue",
        ascending=False
    ).reset_index(drop=True)

    total_revenue = result["total_revenue"].sum()

    if total_revenue == 0:
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
# 8. XYZ Analysis
# ============================================================

def add_xyz_class(df):
    result = df.copy()

    # CV thấp nghĩa là nhu cầu ổn định hơn.
    # Dùng phân vị để nhóm phù hợp với dữ liệu thực tế.
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
# 9. Tính điểm ưu tiên đa tiêu chí
# ============================================================

def add_priority_score(df):
    result = df.copy()

    result["revenue_score"] = minmax(result["total_revenue"])
    result["demand_score"] = minmax(result["total_quantity_sold"])
    result["profit_score"] = minmax(result["estimated_profit"])
    result["damage_risk_score"] = minmax(result["damage_rate"])
    result["variability_risk_score"] = minmax(result["demand_cv"])

    # Hạn sử dụng càng ngắn thì rủi ro càng cao.
    result["perishability_risk_score"] = 1 - minmax(result["shelf_life_days"])

    result["priority_score"] = (
        0.25 * result["revenue_score"]
        + 0.20 * result["demand_score"]
        + 0.20 * result["profit_score"]
        + 0.15 * result["damage_risk_score"]
        + 0.10 * result["perishability_risk_score"]
        + 0.10 * result["variability_risk_score"]
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

    result["high_damage_flag"] = (
        result["damage_rate"] >= result["damage_rate"].quantile(0.75)
    )

    result["high_perishability_flag"] = (
        result["shelf_life_days"] <= result["shelf_life_days"].quantile(0.25)
    )

    result["high_variability_flag"] = (
        result["demand_cv"] >= result["demand_cv"].quantile(0.75)
    )

    return result


# ============================================================
# 10. Khuyến nghị hành động
# ============================================================

def recommend_action(row):
    abc = row["abc_class"]
    xyz = row["xyz_class"]

    damage = row["high_damage_flag"]
    perish = row["high_perishability_flag"]
    shortage = row["stock_shortage_amount"] > 0
    overstock = row["overstock_amount"] > 0

    if abc == "A" and (xyz == "Z" or damage or perish):
        return (
            "Critical control: ưu tiên theo dõi sát, nhập theo lô nhỏ, "
            "kiểm soát hư hỏng và biến động nhu cầu."
        )

    if abc == "A" and xyz in ["X", "Y"]:
        return (
            "High replenishment priority: ưu tiên đảm bảo hàng sẵn có "
            "vì sản phẩm đóng góp doanh thu cao."
        )

    if shortage:
        return (
            "Replenishment warning: kiểm tra nguy cơ thiếu hàng "
            "so với mức tồn kho tối thiểu."
        )

    if overstock and (damage or perish):
        return (
            "Overstock risk: giảm nhập bổ sung và ưu tiên bán/xả tồn "
            "để hạn chế hư hỏng."
        )

    if abc == "C" and xyz == "Z":
        return (
            "Low priority or rationalize: nhu cầu thấp và biến động cao, "
            "không nên nhập dư."
        )

    if damage or perish:
        return (
            "Monitor spoilage: duy trì mức tồn vừa phải và theo dõi hư hỏng."
        )

    return "Standard control: duy trì chính sách tồn kho thông thường."


def add_decision_rules(df):
    result = df.copy()
    result["recommended_action"] = result.apply(recommend_action, axis=1)
    return result


# ============================================================
# 11. KMeans clustering sản phẩm
# ============================================================

def add_kmeans_clusters(df):
    result = df.copy()

    cluster_features = pd.DataFrame({
        "log_total_revenue": np.log1p(result["total_revenue"]),
        "log_total_quantity_sold": np.log1p(result["total_quantity_sold"]),
        "log_estimated_profit": np.log1p(result["estimated_profit"].clip(lower=0)),
        "damage_rate": result["damage_rate"],
        "demand_cv": result["demand_cv"],
        "shelf_life_days": result["shelf_life_days"],
        "margin_percentage": result["margin_percentage"],
        "priority_score": result["priority_score"]
    })

    cluster_features = (
        cluster_features
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_features)

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
            n_init=5,
            max_iter=100,
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
        n_init=5,
        max_iter=100,
        algorithm="lloyd"
    )

    result["kmeans_cluster"] = final_model.fit_predict(X)

    silhouette_df = pd.DataFrame(silhouette_rows)

    silhouette_df.to_csv(
        os.path.join(OUTPUT_DIR, "inventory_kmeans_silhouette_scores.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    cluster_profile = (
        result
        .groupby("kmeans_cluster")
        .agg(
            n_products=("product_id", "count"),
            avg_priority_score=("priority_score", "mean"),
            avg_revenue=("total_revenue", "mean"),
            avg_quantity_sold=("total_quantity_sold", "mean"),
            avg_profit=("estimated_profit", "mean"),
            avg_damage_rate=("damage_rate", "mean"),
            avg_demand_cv=("demand_cv", "mean"),
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

    cluster_profile.to_csv(
        os.path.join(OUTPUT_DIR, "inventory_kmeans_cluster_profile.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return result, silhouette_df, cluster_profile, best_k, best_score


# ============================================================
# 12. Tạo bảng kết quả
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
        "min_stock_level",
        "max_stock_level",
        "total_quantity_sold",
        "total_revenue",
        "estimated_profit",
        "mean_monthly_demand",
        "std_monthly_demand",
        "demand_cv",
        "total_stock_received",
        "total_damaged_stock",
        "damage_rate",
        "estimated_net_stock",
        "stock_shortage_amount",
        "overstock_amount",
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
        os.path.join(OUTPUT_DIR, "product_inventory_decision_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    top_critical = decision_table.head(30)

    top_critical.to_csv(
        os.path.join(OUTPUT_DIR, "top_critical_inventory_products.csv"),
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
            avg_damage_rate=("damage_rate", "mean"),
            avg_priority_score=("priority_score", "mean")
        )
        .reset_index()
        .sort_values(["abc_class", "xyz_class"])
    )

    abc_xyz_summary.to_csv(
        os.path.join(OUTPUT_DIR, "abc_xyz_summary.csv"),
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
            avg_margin=("margin_percentage", "mean"),
            avg_damage_rate=("damage_rate", "mean"),
            avg_priority_score=("priority_score", "mean"),
            critical_products=("priority_level", lambda x: int((x == "Critical").sum()))
        )
        .reset_index()
        .sort_values("avg_priority_score", ascending=False)
    )

    category_summary.to_csv(
        os.path.join(OUTPUT_DIR, "category_inventory_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    priority_summary = (
        df
        .groupby("priority_level")
        .agg(
            n_products=("product_id", "count"),
            avg_revenue=("total_revenue", "mean"),
            avg_profit=("estimated_profit", "mean"),
            avg_damage_rate=("damage_rate", "mean"),
            avg_priority_score=("priority_score", "mean")
        )
        .reset_index()
    )

    priority_summary.to_csv(
        os.path.join(OUTPUT_DIR, "priority_level_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return decision_table, top_critical, abc_xyz_summary, category_summary, priority_summary


# ============================================================
# 13. Bảng mô tả mô hình theo checklist
# ============================================================

def create_model_detail_table(best_k, best_silhouette):
    rows = [
        {
            "component": "ABC Analysis",
            "model_type": "Rule-based inventory classification",
            "input_features": "total_revenue per product",
            "output": "abc_class: A, B, C",
            "stopping_condition": "Không có huấn luyện lặp. Dừng sau khi tính tỷ trọng doanh thu tích lũy cho toàn bộ sản phẩm.",
            "hyperparameter_optimization": "Không tối ưu siêu tham số. Ngưỡng nghiệp vụ dùng chuẩn ABC: A đến 80%, B đến 95%, C còn lại.",
            "main_parameters": "A: cumulative revenue <= 80%; B: <= 95%; C: > 95%",
            "evaluation": "Kiểm tra phân bố số sản phẩm và tỷ trọng doanh thu theo từng nhóm ABC.",
            "business_role": "Xác định sản phẩm đóng góp doanh thu quan trọng nhất để ưu tiên quản trị tồn kho."
        },
        {
            "component": "XYZ Analysis",
            "model_type": "Rule-based demand variability classification",
            "input_features": "monthly_quantity_sold, mean_monthly_demand, std_monthly_demand, demand_cv",
            "output": "xyz_class: X, Y, Z",
            "stopping_condition": "Không có huấn luyện lặp. Dừng sau khi tính hệ số biến động nhu cầu cho từng sản phẩm.",
            "hyperparameter_optimization": "Không tối ưu siêu tham số. Dữ liệu được chia theo phân vị của demand_cv để bảo đảm phân nhóm phù hợp với bộ dữ liệu hiện tại.",
            "main_parameters": "X: nhóm CV thấp nhất; Y: nhóm CV trung bình; Z: nhóm CV cao nhất",
            "evaluation": "Kiểm tra phân bố sản phẩm theo nhóm X, Y, Z và ma trận ABC-XYZ.",
            "business_role": "Đánh giá độ ổn định nhu cầu để quyết định chính sách nhập hàng và mức theo dõi tồn kho."
        },
        {
            "component": "Multi-criteria Priority Scoring",
            "model_type": "Weighted scoring model",
            "input_features": "revenue_score, demand_score, profit_score, damage_risk_score, perishability_risk_score, variability_risk_score",
            "output": "priority_score, priority_level, recommended_action",
            "stopping_condition": "Không có huấn luyện lặp. Dừng sau khi tính điểm ưu tiên cho toàn bộ sản phẩm.",
            "hyperparameter_optimization": "Trọng số được thiết lập theo logic nghiệp vụ của bài toán quản trị tồn kho.",
            "main_parameters": "0.25 revenue, 0.20 demand, 0.20 profit, 0.15 damage, 0.10 perishability, 0.10 variability",
            "evaluation": "Kiểm tra top sản phẩm ưu tiên, phân bố priority_level và tính hợp lý của recommended_action.",
            "business_role": "Tạo điểm ưu tiên dễ diễn giải để hỗ trợ nhà quản lý quyết định sản phẩm nào cần theo dõi trước."
        },
        {
            "component": "KMeans Clustering",
            "model_type": "Unsupervised machine learning clustering",
            "input_features": "log_total_revenue, log_total_quantity_sold, log_estimated_profit, damage_rate, demand_cv, shelf_life_days, margin_percentage, priority_score",
            "output": "kmeans_cluster, cluster_priority_rank",
            "stopping_condition": "Dừng khi thuật toán KMeans hội tụ hoặc đạt số vòng lặp tối đa nội bộ.",
            "hyperparameter_optimization": "Thử k từ 3 đến 6 và chọn k có silhouette_score cao nhất.",
            "main_parameters": f"best_k={best_k}, best_silhouette_score={best_silhouette:.4f}, random_state=42, n_init=5",
            "evaluation": "Silhouette score và bảng cluster profile theo doanh thu, nhu cầu, lợi nhuận, hư hỏng, độ biến động và điểm ưu tiên.",
            "business_role": "Phát hiện các nhóm sản phẩm có hành vi tồn kho tương tự để hỗ trợ chính sách quản trị theo nhóm."
        }
    ]

    detail = pd.DataFrame(rows)

    detail.to_csv(
        os.path.join(OUTPUT_DIR, "inventory_model_detail_checklist.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return detail


# ============================================================
# 14. Vẽ hình
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


def create_figures(df, category_summary, top_critical, cluster_profile):
    abc_counts = df["abc_class"].value_counts().sort_index()

    save_bar_chart(
        abc_counts,
        "Phân bố sản phẩm theo nhóm ABC",
        "ABC class",
        "Số sản phẩm",
        "abc_distribution.png"
    )

    xyz_counts = df["xyz_class"].value_counts().sort_index()

    save_bar_chart(
        xyz_counts,
        "Phân bố sản phẩm theo nhóm XYZ",
        "XYZ class",
        "Số sản phẩm",
        "xyz_distribution.png"
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
        "priority_level_distribution.png"
    )

    # Ma trận ABC-XYZ
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
    plt.savefig(os.path.join(FIGURE_DIR, "abc_xyz_matrix.png"), dpi=300)
    plt.close()

    # Top sản phẩm ưu tiên
    top_plot = top_critical.head(15).sort_values("priority_score")

    labels = (
        top_plot["product_name"].astype(str)
        + " - "
        + top_plot["product_id"].astype(str)
    )

    plt.figure(figsize=(11, 7))
    plt.barh(labels, top_plot["priority_score"])
    plt.title("Top 15 sản phẩm có điểm ưu tiên tồn kho cao nhất")
    plt.xlabel("Priority score")
    plt.ylabel("Sản phẩm")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "top_priority_products.png"), dpi=300)
    plt.close()

    # Damage rate theo category
    damage_plot = category_summary.sort_values(
        "avg_damage_rate",
        ascending=False
    )

    plt.figure(figsize=(11, 6))
    plt.bar(damage_plot["category"], damage_plot["avg_damage_rate"])
    plt.title("Tỷ lệ hàng hỏng trung bình theo danh mục")
    plt.xlabel("Danh mục")
    plt.ylabel("Average damage rate")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "damage_rate_by_category.png"), dpi=300)
    plt.close()

    # Priority score distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["priority_score"], bins=20)
    plt.title("Phân phối điểm ưu tiên quản trị tồn kho")
    plt.xlabel("Priority score")
    plt.ylabel("Số sản phẩm")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "priority_score_distribution.png"), dpi=300)
    plt.close()

    # Cluster profile
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
    plt.savefig(os.path.join(FIGURE_DIR, "kmeans_cluster_priority_profile.png"), dpi=300)
    plt.close()


# ============================================================
# 15. Main
# ============================================================

def main():
    start_time = time.time()

    products, order_items, orders, inventory = load_data()

    quality_summary = create_data_quality_summary(
        products,
        order_items,
        orders,
        inventory
    )

    product_df, monthly_demand = build_product_level_dataset(
        products,
        order_items,
        orders,
        inventory
    )

    product_df = add_abc_class(product_df)
    product_df = add_xyz_class(product_df)
    product_df = add_priority_score(product_df)
    product_df = add_decision_rules(product_df)

    product_df, silhouette_df, cluster_profile, best_k, best_silhouette = (
        add_kmeans_clusters(product_df)
    )

    (
        decision_table,
        top_critical,
        abc_xyz_summary,
        category_summary,
        priority_summary
    ) = create_summary_tables(product_df)

    model_detail = create_model_detail_table(best_k, best_silhouette)

    monthly_demand.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_product_demand_panel.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    create_figures(
        product_df,
        category_summary,
        top_critical,
        cluster_profile
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 90)
    print("HOÀN THÀNH BÀI TOÁN HỖ TRỢ QUYẾT ĐỊNH QUẢN TRỊ TỒN KHO")
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

    print("\nTop 10 sản phẩm cần ưu tiên quản trị tồn kho:")

    cols_to_print = [
        "product_id",
        "product_name",
        "category",
        "total_revenue",
        "total_quantity_sold",
        "damage_rate",
        "demand_cv",
        "abc_class",
        "xyz_class",
        "priority_score",
        "priority_level"
    ]

    print(
        top_critical[cols_to_print]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\nCác file kết quả đã lưu trong thư mục outputs.")
    print("Các hình đã lưu trong thư mục figures.")
    print(f"Thời gian chạy: {elapsed:.2f} giây")


if __name__ == "__main__":
    main()