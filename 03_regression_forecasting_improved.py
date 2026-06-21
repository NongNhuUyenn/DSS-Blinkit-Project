import os
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. Cấu hình
# ============================================================

ORDERS_PATH = "blinkit_orders_clean.csv"

OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. Hàm đánh giá hồi quy
# ============================================================

def regression_metrics(y_true, y_pred):
    """
    Tính các tiêu chí đánh giá hồi quy trên thang đo gốc.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_pred = np.maximum(y_pred, 0)

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
# 3. Tạo chuỗi thời gian theo ngày
# ============================================================

def create_daily_series(orders, target_name):
    """
    Chuyển dữ liệu đơn hàng thành chuỗi thời gian theo ngày.
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

    daily["target"] = daily["target"].fillna(0)

    return daily


# ============================================================
# 4. Tạo đặc trưng nâng cao
# ============================================================

def add_time_series_features(daily):
    """
    Tạo đặc trưng thời gian, biến trễ, trung bình trượt và đặc trưng chu kỳ.
    """

    df = daily.copy()

    df["day_index"] = np.arange(len(df))
    df["day_of_week"] = df["order_day"].dt.dayofweek
    df["month"] = df["order_day"].dt.month
    df["day_of_month"] = df["order_day"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Đặc trưng chu kỳ
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Biến trễ
    lag_list = [1, 2, 3, 7, 14, 21, 28]
    for lag in lag_list:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    # Rolling features
    for window in [7, 14, 28]:
        shifted = df["target"].shift(1)

        df[f"rolling_mean_{window}"] = shifted.rolling(window=window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window=window).std()
        df[f"rolling_min_{window}"] = shifted.rolling(window=window).min()
        df[f"rolling_max_{window}"] = shifted.rolling(window=window).max()
        df[f"rolling_median_{window}"] = shifted.rolling(window=window).median()

    # Expanding mean chỉ dùng dữ liệu quá khứ
    df["expanding_mean"] = df["target"].shift(1).expanding().mean()

    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# 5. Chia train/test theo thời gian
# ============================================================

def time_train_test_split(df, test_ratio=0.2):
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


# ============================================================
# 6. Baseline chuỗi thời gian
# ============================================================

def baseline_predict(test_df, baseline_name):
    """
    Các baseline không cần huấn luyện.
    """

    if baseline_name == "Naive Forecast":
        return test_df["lag_1"].values

    if baseline_name == "Seasonal Naive 7 days":
        return test_df["lag_7"].values

    if baseline_name == "Moving Average 7 days":
        return test_df["rolling_mean_7"].values

    if baseline_name == "Moving Average 14 days":
        return test_df["rolling_mean_14"].values

    if baseline_name == "Moving Average 28 days":
        return test_df["rolling_mean_28"].values

    raise ValueError("Unknown baseline name")


# ============================================================
# 7. Danh sách mô hình và không gian siêu tham số
# ============================================================

def get_feature_columns(df):
    ignore_cols = ["order_day", "target"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    return feature_cols


def build_preprocessor(feature_cols, scale_numeric):
    if scale_numeric:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), feature_cols)
            ],
            remainder="drop"
        )

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols)
        ],
        remainder="drop"
    )


def get_tuned_model_specs(feature_cols):
    """
    Tạo danh sách mô hình cần tối ưu siêu tham số.
    """

    scaled_preprocess = build_preprocessor(feature_cols, scale_numeric=True)
    tree_preprocess = build_preprocessor(feature_cols, scale_numeric=False)

    specs = []

    specs.append({
        "name": "Ridge Regression Tuned",
        "pipeline": Pipeline([
            ("preprocess", scaled_preprocess),
            ("model", Ridge(random_state=RANDOM_STATE))
        ]),
        "param_distributions": {
            "model__alpha": np.logspace(-3, 3, 30)
        },
        "n_iter": 20
    })

    specs.append({
        "name": "Lasso Regression Tuned",
        "pipeline": Pipeline([
            ("preprocess", scaled_preprocess),
            ("model", Lasso(max_iter=20000, random_state=RANDOM_STATE))
        ]),
        "param_distributions": {
            "model__alpha": np.logspace(-4, 1, 30)
        },
        "n_iter": 20
    })

    specs.append({
        "name": "Random Forest Regressor Tuned",
        "pipeline": Pipeline([
            ("preprocess", tree_preprocess),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))
        ]),
        "param_distributions": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", 0.5, 0.8, 1.0]
        },
        "n_iter": 25
    })

    specs.append({
        "name": "Extra Trees Regressor Tuned",
        "pipeline": Pipeline([
            ("preprocess", tree_preprocess),
            ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1))
        ]),
        "param_distributions": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_leaf": [1, 2, 3, 5, 8],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", 0.5, 0.8, 1.0]
        },
        "n_iter": 25
    })

    specs.append({
        "name": "Gradient Boosting Regressor Tuned",
        "pipeline": Pipeline([
            ("preprocess", tree_preprocess),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
        ]),
        "param_distributions": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [1, 3, 5, 8],
            "model__subsample": [0.7, 0.85, 1.0]
        },
        "n_iter": 25
    })

    specs.append({
        "name": "HistGradientBoosting Regressor Tuned",
        "pipeline": Pipeline([
            ("preprocess", tree_preprocess),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))
        ]),
        "param_distributions": {
            "model__max_iter": [100, 200, 300],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            "model__min_samples_leaf": [10, 20, 30]
        },
        "n_iter": 25
    })

    return specs


# ============================================================
# 8. Tuning bằng TimeSeriesSplit
# ============================================================

def tune_and_fit_model(spec, X_train, y_train, n_splits=5):
    """
    Tối ưu siêu tham số bằng TimeSeriesSplit.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=spec["pipeline"],
        param_distributions=spec["param_distributions"],
        n_iter=spec["n_iter"],
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, -search.best_score_


# ============================================================
# 9. Chạy thí nghiệm cho một biến mục tiêu
# ============================================================

def run_improved_forecasting_experiment(orders, target_name, display_name):
    print("\n" + "=" * 90)
    print(f"ĐANG CHẠY CẢI TIẾN MÔ HÌNH: {display_name}")
    print("=" * 90)

    raw_daily = create_daily_series(orders, target_name)
    df = add_time_series_features(raw_daily)

    train_df, test_df = time_train_test_split(df, test_ratio=0.2)

    feature_cols = get_feature_columns(df)

    X_train = train_df[feature_cols]
    y_train_original = train_df["target"].values

    X_test = test_df[feature_cols]
    y_test_original = test_df["target"].values

    result_rows = []
    prediction_rows = []
    model_detail_rows = []

    best_mae = np.inf
    best_model_name = None
    best_predictions = None

    # ------------------------------------------------------------
    # 9.1. Chạy baseline
    # ------------------------------------------------------------

    baseline_names = [
        "Naive Forecast",
        "Seasonal Naive 7 days",
        "Moving Average 7 days",
        "Moving Average 14 days",
        "Moving Average 28 days"
    ]

    for baseline_name in baseline_names:
        print(f"Running baseline: {baseline_name}")

        start_pred = time.time()
        y_pred = baseline_predict(test_df, baseline_name)
        predict_time = time.time() - start_pred

        metrics = regression_metrics(y_test_original, y_pred)

        row = {
            "target": target_name,
            "target_display": display_name,
            "model": baseline_name,
            "model_group": "Baseline",
            "target_transform": "None",
            **metrics,
            "cv_best_mae": np.nan,
            "train_time_seconds": 0.0,
            "predict_time_seconds": predict_time,
            "best_params": "{}"
        }

        result_rows.append(row)

        pred_df = pd.DataFrame({
            "target": target_name,
            "target_display": display_name,
            "order_day": test_df["order_day"],
            "model": baseline_name,
            "actual": y_test_original,
            "predicted": np.maximum(y_pred, 0),
            "error": y_test_original - np.maximum(y_pred, 0),
            "absolute_error": np.abs(y_test_original - np.maximum(y_pred, 0))
        })

        prediction_rows.append(pred_df)

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_model_name = baseline_name
            best_predictions = pred_df.copy()

        model_detail_rows.append({
            "target": target_name,
            "model": baseline_name,
            "model_type": "Time-series baseline",
            "stopping_condition": "Không có quá trình huấn luyện. Dự báo được tính trực tiếp từ giá trị quá khứ.",
            "hyperparameter_optimization": "Không tối ưu siêu tham số.",
            "main_hyperparameters": baseline_name,
            "test_result_main": f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, NMAE={metrics['NMAE']:.4f}, R2={metrics['R2']:.4f}",
            "comment": "Dùng làm mốc so sánh để kiểm tra mô hình học máy có thật sự cải thiện so với quy tắc dự báo đơn giản hay không."
        })

    # ------------------------------------------------------------
    # 9.2. Chạy mô hình học máy có tuning
    # ------------------------------------------------------------

    specs = get_tuned_model_specs(feature_cols)

    # Với doanh thu, thử cả mô hình trên target gốc và target log.
    # Với số đơn, chỉ dùng target gốc để tránh làm phức tạp diễn giải.
    target_transforms = ["None"]

    if target_name == "daily_revenue":
        target_transforms.append("log1p")

    for transform in target_transforms:
        if transform == "None":
            y_train = y_train_original.copy()
        elif transform == "log1p":
            y_train = np.log1p(y_train_original)
        else:
            raise ValueError("Unknown target transform")

        for spec in specs:
            model_name = spec["name"]

            if transform == "log1p":
                model_name = model_name + " with log target"

            print(f"Tuning model: {model_name}")

            start_train = time.time()
            best_estimator, best_params, cv_best_mae = tune_and_fit_model(
                spec=spec,
                X_train=X_train,
                y_train=y_train,
                n_splits=5
            )
            train_time = time.time() - start_train

            start_pred = time.time()
            y_pred_model_scale = best_estimator.predict(X_test)

            if transform == "log1p":
                y_pred = np.expm1(y_pred_model_scale)
            else:
                y_pred = y_pred_model_scale

            y_pred = np.maximum(y_pred, 0)
            predict_time = time.time() - start_pred

            metrics = regression_metrics(y_test_original, y_pred)

            row = {
                "target": target_name,
                "target_display": display_name,
                "model": model_name,
                "model_group": "Tuned ML",
                "target_transform": transform,
                **metrics,
                "cv_best_mae": cv_best_mae,
                "train_time_seconds": train_time,
                "predict_time_seconds": predict_time,
                "best_params": str(best_params)
            }

            result_rows.append(row)

            pred_df = pd.DataFrame({
                "target": target_name,
                "target_display": display_name,
                "order_day": test_df["order_day"],
                "model": model_name,
                "actual": y_test_original,
                "predicted": y_pred,
                "error": y_test_original - y_pred,
                "absolute_error": np.abs(y_test_original - y_pred)
            })

            prediction_rows.append(pred_df)

            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                best_model_name = model_name
                best_predictions = pred_df.copy()

            model_detail_rows.append({
                "target": target_name,
                "model": model_name,
                "model_type": "Tuned regression model",
                "stopping_condition": get_stopping_condition(model_name),
                "hyperparameter_optimization": "RandomizedSearchCV với TimeSeriesSplit, tiêu chí tối ưu là MAE trên các lát cắt thời gian.",
                "main_hyperparameters": str(best_params),
                "test_result_main": f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, NMAE={metrics['NMAE']:.4f}, R2={metrics['R2']:.4f}",
                "comment": get_model_comment(model_name)
            })

    results_df = pd.DataFrame(result_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    model_detail_df = pd.DataFrame(model_detail_rows)

    results_df = results_df.sort_values("MAE").reset_index(drop=True)

    safe_target_name = target_name.replace("daily_", "")

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{safe_target_name}_improved_regression_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{safe_target_name}_improved_predictions_all_models.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    model_detail_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{safe_target_name}_model_detail_checklist.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    top_errors = (
        best_predictions
        .sort_values("absolute_error", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    top_errors.to_csv(
        os.path.join(OUTPUT_DIR, f"{safe_target_name}_improved_top_error_days.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 9.3. Vẽ ảnh
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 5))
    plt.plot(raw_daily["order_day"], raw_daily["target"])
    plt.title(f"Chuỗi thời gian gốc - {display_name}")
    plt.xlabel("Ngày")
    plt.ylabel(display_name)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_daily_series_improved.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(best_predictions["order_day"], best_predictions["actual"], label="Thực tế")
    plt.plot(best_predictions["order_day"], best_predictions["predicted"], label="Dự báo")
    plt.title(f"So sánh thực tế và dự báo - {display_name} - {best_model_name}")
    plt.xlabel("Ngày")
    plt.ylabel(display_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_improved_actual_vs_predicted.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(best_predictions["predicted"], best_predictions["error"])
    plt.axhline(0, linestyle="--")
    plt.title(f"Biểu đồ phần dư - {display_name} - {best_model_name}")
    plt.xlabel("Giá trị dự báo")
    plt.ylabel("Sai số thực tế trừ dự báo")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_improved_residual_plot.png"),
        dpi=300
    )
    plt.close()

    plot_df = results_df.head(12).sort_values("MAE")

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, plot_df["MAE"], width, label="MAE")
    plt.bar(x + width / 2, plot_df["RMSE"], width, label="RMSE")
    plt.xticks(x, plot_df["model"], rotation=45, ha="right")
    plt.title(f"So sánh mô hình theo MAE và RMSE - {display_name}")
    plt.ylabel("Sai số")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, f"{safe_target_name}_improved_model_comparison.png"),
        dpi=300
    )
    plt.close()

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
        os.path.join(FIGURE_DIR, f"{safe_target_name}_improved_top_error_days.png"),
        dpi=300
    )
    plt.close()

    print("\nKết quả tốt nhất:")
    print(results_df.head(10).round(4).to_string(index=False))
    print("\nMô hình tốt nhất:", best_model_name)
    print("MAE tốt nhất:", best_mae)

    return results_df, predictions_df, model_detail_df, top_errors, best_model_name


# ============================================================
# 10. Mô tả điều kiện dừng và chú giải
# ============================================================

def get_stopping_condition(model_name):
    if "Ridge" in model_name:
        return "Dừng khi nghiệm tối ưu của hàm mất mát có điều chuẩn L2 được tìm thấy."
    if "Lasso" in model_name:
        return "Dừng khi nghiệm hội tụ hoặc đạt số vòng lặp tối đa."
    if "Random Forest" in model_name:
        return "Dừng khi xây đủ số lượng cây được chọn trong quá trình tối ưu siêu tham số."
    if "Extra Trees" in model_name:
        return "Dừng khi xây đủ số lượng cây trong ensemble."
    if "Gradient Boosting" in model_name and "Hist" not in model_name:
        return "Dừng khi hoàn thành số vòng boosting được chọn."
    if "HistGradientBoosting" in model_name:
        return "Dừng khi đạt số vòng lặp tối đa hoặc điều kiện hội tụ nội bộ của thuật toán."
    return "Dừng theo điều kiện hội tụ hoặc số vòng lặp tối đa của mô hình."


def get_model_comment(model_name):
    if "Ridge" in model_name:
        return "Mô hình hồi quy tuyến tính có điều chuẩn L2, giúp giảm độ lớn hệ số và hạn chế overfitting."
    if "Lasso" in model_name:
        return "Mô hình hồi quy tuyến tính có điều chuẩn L1, có khả năng co một số hệ số về 0."
    if "Random Forest" in model_name:
        return "Mô hình ensemble nhiều cây, có khả năng học quan hệ phi tuyến và tương tác giữa đặc trưng."
    if "Extra Trees" in model_name:
        return "Mô hình ensemble nhiều cây với mức ngẫu nhiên cao hơn, có thể giảm phương sai."
    if "Gradient Boosting" in model_name and "Hist" not in model_name:
        return "Mô hình boosting tuần tự, mỗi cây sau tập trung sửa lỗi của các cây trước."
    if "HistGradientBoosting" in model_name:
        return "Mô hình boosting dựa trên histogram, phù hợp với dữ liệu bảng và có tốc độ huấn luyện tốt."
    return "Mô hình dùng để so sánh trong nhóm hồi quy."


# ============================================================
# 11. Tạo bảng tổng hợp cải tiến
# ============================================================

def create_improvement_summary(all_results):
    """
    Tạo bảng so sánh phiên bản cũ và phiên bản cải tiến theo mô hình tốt nhất.
    """

    best_by_target = (
        all_results
        .sort_values("MAE")
        .groupby("target")
        .head(1)
        .reset_index(drop=True)
    )

    best_by_target.to_csv(
        os.path.join(OUTPUT_DIR, "regression_best_models_after_improvement.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return best_by_target


# ============================================================
# 12. Main
# ============================================================

def main():
    orders = pd.read_csv(
        ORDERS_PATH,
        parse_dates=["order_date", "promised_delivery_time", "actual_delivery_time"]
    )

    revenue_results, revenue_predictions, revenue_detail, revenue_top_errors, revenue_best = run_improved_forecasting_experiment(
        orders=orders,
        target_name="daily_revenue",
        display_name="Doanh thu ngày"
    )

    orders_results, orders_predictions, orders_detail, orders_top_errors, orders_best = run_improved_forecasting_experiment(
        orders=orders,
        target_name="daily_orders",
        display_name="Số lượng đơn hàng ngày"
    )

    all_results = pd.concat(
        [revenue_results, orders_results],
        ignore_index=True
    )

    all_details = pd.concat(
        [revenue_detail, orders_detail],
        ignore_index=True
    )

    all_results.to_csv(
        os.path.join(OUTPUT_DIR, "all_improved_regression_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_details.to_csv(
        os.path.join(OUTPUT_DIR, "all_regression_model_detail_checklist.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    best_summary = create_improvement_summary(all_results)

    print("\n" + "=" * 90)
    print("HOÀN THÀNH CẢI TIẾN MÔ HÌNH HỒI QUY")
    print("=" * 90)

    print("\nMô hình tốt nhất sau cải tiến:")
    print(best_summary.round(4).to_string(index=False))

    print("\nFile kết quả đã lưu trong thư mục outputs.")
    print("Hình ảnh đã lưu trong thư mục figures.")


if __name__ == "__main__":
    main()