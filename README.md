# DSS Blinkit Project

This repository contains the implementation for Member 3's part of the DSS Blinkit project.

## Main Objectives

This part focuses on two decision-support tasks:

1. Forecasting daily revenue and daily order volume.
2. Predicting delivery delay risk for each order.

## Input Data

The project uses the following datasets:

- `blinkit_orders_clean.csv`
- `blinkit_marketing_performance_clean.csv`
- `blinkit_data2/blinkit_order_items.csv`
- `blinkit_data2/blinkit_products.csv`
- `blinkit_data2/blinkit_customers.csv`
- `blinkit_data2/blinkit_delivery_performance.csv`

## Main Notebook

The main notebook is:

- `member3_forecast_prediction_ready_fixed.ipynb`

## Outputs

The notebook generates the following output files:

- `outputs/daily_forecast_model_comparison.csv`
- `outputs/daily_revenue_predictions.csv`
- `outputs/daily_orders_predictions.csv`
- `outputs/future_14_days_forecast.csv`
- `outputs/delivery_model_comparison.csv`
- `outputs/delivery_risk_recommendations.csv`
- `outputs/decision_tree_rules.txt`
- `outputs/master_order_data_for_member3.csv`

## Figures

Generated figures are stored in the `figures/` folder, including:

- Daily revenue trend
- Daily order volume trend
- Forecast results on the test period
- Future 14-day revenue forecast
- Future 14-day order volume forecast

## Models Used

### Forecasting Models

- 7-day Moving Average
- Linear Regression
- Random Forest Regressor

### Delivery Delay Prediction Models

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- SVM

## Decision Support Meaning

The forecasting results help managers prepare inventory and operational resources for future demand.

The delivery delay prediction results help managers identify high-risk orders and prioritize them for operational handling.
