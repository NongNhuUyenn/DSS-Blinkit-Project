# DSS Blinkit Project

This repository contains the source code, experimental outputs, figures, and supporting materials for the Decision Support System course project using Blinkit retail data.

## Project Overview

This project builds a data-driven decision support system for product portfolio management. Instead of only describing historical sales data, the project focuses on supporting managers in identifying products that should be prioritized for monitoring in the next business period.

The main decision problem is:

> Which products are likely to belong to the high future performance group in the next three months?

To answer this question, the project converts historical Blinkit sales data into a supervised machine learning problem at the product-time level. Each product is evaluated using historical sales behavior, product attributes, and category information. The model then predicts whether the product is likely to be in the top 25% of future three-month revenue.

## Main Components

The project includes:

* Data preprocessing and exploratory analysis
* PowerBI-based business analysis
* Product portfolio decision support analysis
* Regression forecasting experiments
* Product future performance classification
* Extended benchmark of multiple machine learning models
* Model evaluation, error analysis, and recommendation tables

## Main Machine Learning Task

The final machine learning task is a binary classification problem.

For each product at a cutoff time, the model predicts whether the product will belong to the high future performance group in the next three months.

The target variable is:

`high_future_performance`

A product is labeled as high future performance if its future three-month revenue is in the top 25% among products at the same cutoff time.


## Main Files

The main file for the final machine learning benchmark is:

```text
08_product_future_performance_extended_benchmark.py
````

This file is the final benchmark script used to build the supervised product future performance dataset, train multiple classification models, evaluate model performance, generate result tables, and export figures for the report.

Other scripts are earlier experimental or supporting versions:

* `02_regression_forecasting_models.py`: initial regression forecasting experiments
* `03_regression_forecasting_improved.py`: improved regression forecasting experiments
* `04_inventory_decision_support.py`: inventory-related decision support experiment
* `04_product_portfolio_decision_support.py`: product portfolio analysis experiment
* `05_product_future_performance_classification.py`: first product classification version
* `06_product_future_performance_classification_final.py`: refined product classification version
* `07_product_future_performance_classification_tabpfn.py`: TabPFN-related experimental version
* `08_product_future_performance_extended_benchmark.py`: final main benchmark script
* `member3_forecast_prediction_ready_fixed.ipynb`: notebook used for testing and demonstration

```

## Data Used

The main model uses cleaned Blinkit sales data, including:

* Orders data
* Order items data
* Product information data

Inventory, marketing, customer feedback, and delivery data are not used in the final main classification model when they are not directly required or not consistent enough for the decision problem.

## Models

The project benchmarks multiple classification models, including:

* Logistic Regression Balanced
* Linear SVM Balanced
* Decision Tree Balanced
* Random Forest Balanced
* Extra Trees Balanced
* Gradient Boosting
* HistGradientBoosting
* Gaussian Naive Bayes
* KNN Classifier
* MLP Classifier
* LightGBM Balanced
* XGBoost Balanced
* CatBoost Balanced

## Best Model

The selected final model is:

`Random Forest Balanced`

Main test results:

| Metric            |  Value |
| ----------------- | -----: |
| Accuracy          | 0.8134 |
| Balanced Accuracy | 0.7794 |
| Precision         | 0.6085 |
| Recall            | 0.7114 |
| F1-score          | 0.6560 |
| ROC-AUC           | 0.8742 |
| PR-AUC            | 0.6055 |

The model is selected because it achieves the best F1-score and provides a balanced trade-off between detecting high-performing products and limiting false recommendations.

## Project Structure

```text
DSS_Blinkit_Project/
├── 02_regression_forecasting_models.py
├── 03_regression_forecasting_improved.py
├── 04_inventory_decision_support.py
├── 04_product_portfolio_decision_support.py
├── 05_product_future_performance_classification.py
├── 06_product_future_performance_classification_final.py
├── 07_product_future_performance_classification_tabpfn.py
├── 08_product_future_performance_extended_benchmark.py
├── member3_forecast_prediction_ready_fixed.ipynb
├── figures/
├── figures_benchmark08/
├── outputs/
├── outputs_benchmark08/
└── README.md
```

## Important Output Files

Main benchmark outputs are stored in:

`outputs_benchmark08/`

Important files include:

* `final_classification_results.csv`
* `final_model_detail_checklist.csv`
* `final_classification_predictions.csv`
* `final_latest_recommendation_table.csv`
* `final_false_positive_cases.csv`
* `final_false_negative_cases.csv`
* `final_best_model_feature_importance.csv`

Main figures are stored in:

`figures_benchmark08/`

Important figures include:

* `final_label_distribution_by_fold.png`
* `final_classification_f1_comparison.png`
* `final_classification_roc_auc_comparison.png`
* `final_best_confusion_matrix.png`
* `final_best_model_roc_curve.png`
* `final_best_model_pr_curve.png`
* `final_best_model_feature_importance.png`

## Decision Support Meaning

The model output is used as a decision support signal, not as an automatic decision rule.

Products with high predicted probability can be prioritized for:

* closer monitoring
* product portfolio review
* inventory planning discussion
* promotion or visibility consideration
* category-level business analysis

The final decision should still combine model results with managerial experience, business strategy, inventory conditions, and operational constraints.

## Course Context

This project is developed for the Decision Support System course.

The report follows an applied research structure:

1. Problem definition
2. Data and exploratory analysis
3. Research methodology
4. Experiments and results
5. Discussion and decision support recommendations
