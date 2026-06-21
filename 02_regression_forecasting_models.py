import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. Cấu hình đường dẫn
# ============================================================

ORDERS_PATH = "blinkit_orders_clean.csv"

OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# 2. Hàm đánh giá hồi quy
# ============================================================

def regression_metrics(y_true, y_pred):
    """
    Tính đầy đủ các chỉ số đánh giá hồi quy.

    y_true: giá trị thực tế.
    y_pred: giá trị dự báo.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mean_y = np.mean(y_true)
    nmae = mae / mean_y if mean_y != 0 else np.nan

    wape = np.sum(np.abs(error)) / np.sum(np.abs(y_true)) if np.sum(np.abs(y_true)) != 0 else np.nan

    r2 = r2_score(y_true, y_pred)

    bias = np.mean(error)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NMAE": nmae,
        "WAPE": wape,
        "R2": r2,
        "Bias_ME": bias
    }


# ============================================================
# 3. Hàm tạo đặc trưng chuỗi thời gian
# ============================================================

def create_daily_series(orders, target_name):
    """
    Tạo dữ liệu chuỗi thời gian theo ngày.

    target_name = "daily_revenue" hoặc "daily_orders".
    """

    orders = orders.copy()
    orders["order_day"] = orders["order_date"].dt.normalize()

    if target_name == "daily_revenue":
        daily = (
            orders
            .groupby("order_day")
            .agg(target=("order_total_clean", "sum"))
            .reset_index()
        )
    elif target_name == "daily_orders":
        daily = (
            orders
            .groupby("order_day")
            .agg(target=("order_id", "count"))
            .reset_index()
        )
    else:
        raise ValueError("target_name must be daily_revenue or daily_orders")

    full_days = pd.date_range(
        daily["order_day"].min(),
        daily["order_day"].max(),
        freq="D"
    )

    daily = (
        daily
        .set_index("order_day")
        .reindex(full_days)
        .rename_axis("order_day")
        .reset_index()
    )

    # Nếu ngày nào không có đơn thì doanh thu và số đơn bằng 0.
    daily["target"] = daily["target"].fillna(0)

    daily["day_index"] = np.arange(len(daily))
    daily["day_of_week"] = daily["order_day"].dt.dayofweek
    daily["month"] = daily["order_day"].dt.month
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    # Biến trễ
    daily["lag_1"] = daily["target"].shift(1)
    daily["lag_7"] = daily["target"].shift(7)
    daily["lag_14"] = daily["target"].shift(14)

    # Trung bình trượt
    daily["rolling_7"] = daily["target"].shift(1).rolling(window=7).mean()
    daily["rolling_14"] = daily["target"].shift(1).rolling(window=14).mean()

    daily = daily.dropna().reset_index(drop=True)

    return daily


# ============================================================
# 4. Hàm chia train/test theo thời gian
# ============================================================

def time_train_test_split(df, test_ratio=0.2):
    """
    Chia train/test theo thứ tự thời gian.
    Không xáo trộn dữ liệu vì đây là bài toán dự báo.
    """

    split_index = int(len(df) * (1 - test_ratio))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


# ============================================================
# 5. Mô hình trung bình động 7 ngày
# ============================================================

def moving_average_predict(test_df):
    """
    Với mô hình Moving Average, dự báo chính là rolling_7 đã tạo từ dữ liệu quá khứ.
    """

    return test_df["rolling_7"].values


# ============================================================
# 6. Danh sách 8 mô hình hồi quy
# ============================================================

def get_regression_models():
    """
    Trả về 8 mô hình hồi quy dùng trong bài toán dự báo.
    """

    numeric_features = [
        "day_index",
        "day_of_week",
        "month",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_7",
        "rolling_14"
    ]

    scaler_preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="drop"
    )

    passthrough_preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features)
        ],
        remainder="drop"
    )

    models = {
        "Moving Average 7 days": None,

        "Linear Regression": Pipeline([
            ("preprocess", scaler_preprocess),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("preprocess", scaler_preprocess),
            ("model", Ridge(alpha=1.0, random_state=42))
        ]),

        "Lasso Regression": Pipeline([
            ("preprocess", scaler_preprocess),
            ("model", Lasso(alpha=0.01, max_iter=10000, random_state=42))
        ]),

        "Random Forest Regressor": Pipeline([
            ("preprocess", passthrough_preprocess),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Extra Trees Regressor": Pipeline([
            ("preprocess", passthrough_preprocess),
            ("model", ExtraTreesRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting Regressor": Pipeline([
            ("preprocess", passthrough_preprocess),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ))
        ]),

        "HistGradientBoosting Regressor": Pipeline([
            ("preprocess", passthrough_preprocess),
            ("model", HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42
            ))
        ])
    }

    return models, numeric_features


# ============================================================
# 7. Hàm chạy 8 mô hình cho một biến mục tiêu
# ============================================================

def run_forecasting_experiment(orders, target_name, display_name):
    """
    Chạy toàn bộ thí nghiệm dự báo cho một biến mục tiêu.
    """

    print("\n" + "=" * 80)
    print(f"Đang chạy bài toán: {display_name}")
    print("=" * 80)

    daily = create_daily_series(orders, target_name)
    train_df, test_df = time_train_test_split(daily, test_ratio=0.2)

    models, feature_cols = get_regression_models()

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    result_rows = []
    prediction_rows = []

    best_model_name = None
    best_mae = np.inf
    best_predictions = None

    for model_name, model in models.items():
        print(f"Training model: {model_name}")

        start_train = time.time()

        if model_name == "Moving Average 7 days":
            train_time = 0.0

            start_pred = time.time()
            y_pred = moving_average_predict(test_df)
            predict_time = time.time() - start_pred

        else:
            model_instance = clone(model)
            model_instance.fit(X_train, y_train)
            train_time = time.time() - start_train

            start_pred = time.time()
            y_pred = model_instance.predict(X_test)
            predict_time = time.time() - start_pred

        metrics = regression_metrics(y_test, y_pred)

        result_row = {
            "target": target_name,
            "target_display": display_name,
            "model": model_name,
            **metrics,
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time
        }

        result_rows.append(result_row)

        temp_pred = pd.DataFrame({
            "target": target_name,
            "target_display": display_name,
            "order_day": test_df["order_day"],
            "model": model_name,
            "actual": y_test.values,
            "predicted": y_pred,
            "error": y_test.values - y_pred,
            "absolute_error": np.abs(y_test.values - y_pred)
        })

        prediction_rows.append(temp_pred)

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_model_name = model_name
            best_predictions = temp_pred.copy()

    results_df = pd.DataFrame(result_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    results_df = results_df.sort_values("MAE").reset_index(drop=True)

    safe_target_name = target_name.replace("daily_", "")

    results_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_target_name}_regression_model_results.csv"
    )

    predictions_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_target_name}_regression_predictions_all_models.csv"
    )

    top_errors_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_target_name}_top_error_days.csv"
    )

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    top_errors = (
        best_predictions
        .sort_values("absolute_error", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    top_errors.to_csv(top_errors_path, index=False, encoding="utf-8-sig")

    print("\nKết quả mô hình:")
    print(results_df)

    print("\nMô hình tốt nhất theo MAE:", best_model_name)
    print("MAE tốt nhất:", best_mae)

    # Vẽ ảnh chuỗi thời gian gốc
    plt.figure(figsize=(12, 5))
    plt.plot(daily["order_day"], daily["target"])
    plt.title(f"{display_name} theo ngày")
    plt.xlabel("Ngày")
    plt.ylabel(display_name)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_daily_series.png"),
        dpi=300
    )
    plt.close()

    # Vẽ actual vs predicted của mô hình tốt nhất
    plt.figure(figsize=(12, 5))
    plt.plot(best_predictions["order_day"], best_predictions["actual"], label="Thực tế")
    plt.plot(best_predictions["order_day"], best_predictions["predicted"], label="Dự báo")
    plt.title(f"So sánh thực tế và dự báo - {display_name} - {best_model_name}")
    plt.xlabel("Ngày")
    plt.ylabel(display_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_actual_vs_predicted.png"),
        dpi=300
    )
    plt.close()

    # Vẽ residual plot
    plt.figure(figsize=(10, 5))
    plt.scatter(best_predictions["predicted"], best_predictions["error"])
    plt.axhline(0, linestyle="--")
    plt.title(f"Biểu đồ phần dư - {display_name} - {best_model_name}")
    plt.xlabel("Giá trị dự báo")
    plt.ylabel("Sai số thực tế trừ dự báo")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_residual_plot.png"),
        dpi=300
    )
    plt.close()

    # Vẽ so sánh MAE và RMSE
    plot_df = results_df.sort_values("MAE")

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, plot_df["MAE"], width, label="MAE")
    plt.bar(x + width / 2, plot_df["RMSE"], width, label="RMSE")
    plt.xticks(x, plot_df["model"], rotation=35, ha="right")
    plt.title(f"So sánh MAE và RMSE - {display_name}")
    plt.ylabel("Sai số")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_model_comparison_mae_rmse.png"),
        dpi=300
    )
    plt.close()

    # Vẽ top 10 ngày lỗi lớn nhất
    top_errors_for_plot = top_errors.sort_values("absolute_error")

    plt.figure(figsize=(10, 5))
    plt.barh(
        top_errors_for_plot["order_day"].dt.strftime("%Y-%m-%d"),
        top_errors_for_plot["absolute_error"]
    )
    plt.title(f"Top 10 ngày có sai số lớn nhất - {display_name}")
    plt.xlabel("Sai số tuyệt đối")
    plt.ylabel("Ngày")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_top_error_days.png"),
        dpi=300
    )
    plt.close()

    return results_df, predictions_df, top_errors, best_model_name


# ============================================================
# 8. Bảng mô tả mô hình theo checklist
# ============================================================

def create_regression_model_description_table():
    """
    Tạo bảng mô tả mô hình theo yêu cầu checklist của thầy.
    """

    rows = [
        {
            "model": "Moving Average 7 days",
            "model_type": "Baseline time-series forecasting",
            "stopping_condition": "Không có quá trình huấn luyện. Dự báo được tính trực tiếp từ trung bình 7 ngày trước đó.",
            "hyperparameter_optimization": "Không tối ưu siêu tham số.",
            "main_hyperparameters": "window = 7",
            "comment": "Dùng làm mô hình cơ sở để kiểm tra các mô hình học máy có cải thiện so với quy tắc đơn giản hay không."
        },
        {
            "model": "Linear Regression",
            "model_type": "Linear regression",
            "stopping_condition": "Dừng khi nghiệm tối ưu của bài toán bình phương tối thiểu được tìm thấy.",
            "hyperparameter_optimization": "Không có siêu tham số chính cần tối ưu trong phiên bản cơ sở.",
            "main_hyperparameters": "fit_intercept = True",
            "comment": "Mô hình tuyến tính cơ sở, dễ giải thích và có tốc độ huấn luyện nhanh."
        },
        {
            "model": "Ridge Regression",
            "model_type": "Regularized linear regression",
            "stopping_condition": "Dừng khi nghiệm tối ưu của hàm mất mát có điều chuẩn L2 được tìm thấy.",
            "hyperparameter_optimization": "Thiết lập alpha cơ sở. Có thể cải tiến bằng GridSearchCV.",
            "main_hyperparameters": "alpha = 1.0",
            "comment": "Giảm hiện tượng hệ số quá lớn, phù hợp khi các đặc trưng có tương quan."
        },
        {
            "model": "Lasso Regression",
            "model_type": "Regularized linear regression",
            "stopping_condition": "Dừng khi nghiệm hội tụ hoặc đạt max_iter.",
            "hyperparameter_optimization": "Thiết lập alpha cơ sở. Có thể cải tiến bằng GridSearchCV.",
            "main_hyperparameters": "alpha = 0.01, max_iter = 10000",
            "comment": "Có khả năng co hệ số và loại bớt đặc trưng ít quan trọng."
        },
        {
            "model": "Random Forest Regressor",
            "model_type": "Bagging ensemble of regression trees",
            "stopping_condition": "Dừng khi xây đủ số cây trong rừng.",
            "hyperparameter_optimization": "Thiết lập thủ công ở mức baseline. Có thể cải tiến bằng RandomizedSearchCV.",
            "main_hyperparameters": "n_estimators = 300, max_depth = 8, min_samples_leaf = 3",
            "comment": "Học được quan hệ phi tuyến và tương tác giữa các đặc trưng."
        },
        {
            "model": "Extra Trees Regressor",
            "model_type": "Randomized tree ensemble",
            "stopping_condition": "Dừng khi xây đủ số cây trong ensemble.",
            "hyperparameter_optimization": "Thiết lập thủ công ở mức baseline. Có thể cải tiến bằng RandomizedSearchCV.",
            "main_hyperparameters": "n_estimators = 300, max_depth = 8, min_samples_leaf = 3",
            "comment": "Tăng mức ngẫu nhiên khi chia cây, thường giúp giảm phương sai."
        },
        {
            "model": "Gradient Boosting Regressor",
            "model_type": "Boosting ensemble of regression trees",
            "stopping_condition": "Dừng khi hoàn thành số vòng boosting.",
            "hyperparameter_optimization": "Thiết lập thủ công ở mức baseline. Có thể cải tiến bằng GridSearchCV.",
            "main_hyperparameters": "n_estimators = 200, learning_rate = 0.05, max_depth = 3",
            "comment": "Xây các cây tuần tự, mỗi cây sau tập trung sửa lỗi của các cây trước."
        },
        {
            "model": "HistGradientBoosting Regressor",
            "model_type": "Histogram-based gradient boosting",
            "stopping_condition": "Dừng khi đạt max_iter hoặc điều kiện hội tụ nội bộ.",
            "hyperparameter_optimization": "Thiết lập thủ công ở mức baseline. Có thể cải tiến bằng RandomizedSearchCV.",
            "main_hyperparameters": "max_iter = 200, learning_rate = 0.05, max_leaf_nodes = 31",
            "comment": "Phiên bản boosting hiệu quả cho dữ liệu bảng, tốc độ tốt hơn khi dữ liệu lớn."
        }
    ]

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTPUT_DIR, "regression_model_description_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return df


# ============================================================
# 9. Chạy toàn bộ thí nghiệm
# ============================================================

def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["order_date", "promised_delivery_time", "actual_delivery_time"]
    )

    revenue_results, revenue_predictions, revenue_top_errors, revenue_best = run_forecasting_experiment(
        orders=orders,
        target_name="daily_revenue",
        display_name="Doanh thu ngày"
    )

    orders_results, orders_predictions, orders_top_errors, orders_best = run_forecasting_experiment(
        orders=orders,
        target_name="daily_orders",
        display_name="Số lượng đơn hàng ngày"
    )

    all_results = pd.concat(
        [revenue_results, orders_results],
        ignore_index=True
    )

    all_results.to_csv(
        os.path.join(OUTPUT_DIR, "all_regression_forecasting_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    description_df = create_regression_model_description_table()

    print("\n" + "=" * 80)
    print("ĐÃ HOÀN THÀNH CHẠY 8 MÔ HÌNH HỒI QUY")
    print("=" * 80)

    print("\nMô hình tốt nhất cho doanh thu theo MAE:", revenue_best)
    print("Mô hình tốt nhất cho số đơn theo MAE:", orders_best)

    print("\nCác file kết quả đã lưu trong thư mục outputs.")
    print("Các hình đã lưu trong thư mục figures.")


if __name__ == "__main__":
    main()