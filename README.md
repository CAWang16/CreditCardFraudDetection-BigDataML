# Credit Card Fraud Detection Using Big Data Machine Learning

**Apache Spark MLlib Baseline with H2O and TensorFlow Comparison**

CSP554 Big Data Technologies — Illinois Institute of Technology

A multi-framework comparison of machine learning pipelines for credit card fraud detection, built for AWS EMR. The same feature engineering and model suite is implemented in three frameworks (Apache Spark MLlib, H2O.ai, and TensorFlow/Keras) to enable apples-to-apples performance comparisons.

Full report: [credit_card_fraud_detection_big_data_ml.pdf](credit_card_fraud_detection_big_data_ml.pdf)

## Dataset

**Credit Card Transactions** CSV stored in S3 (`s3://csp554-project3/credit_card_transactions.csv`).

- Heavily imbalanced: ~0.58% of transactions are fraudulent
- Raw columns include: transaction timestamp, cardholder location, merchant location, transaction amount, cardholder demographics, and merchant category

## Files

| File | Description |
|---|---|
| [fraud_detection_spark.py](fraud_detection_spark.py) | Main Spark MLlib pipeline — runs on AWS EMR via `spark-submit` |
| [test_spark.py](test_spark.py) | Smoke test using a 10,000-row sample from S3 to validate the EMR pipeline |
| [h2o_fraud_from_spark.py](h2o_fraud_from_spark.py) | H2O.ai equivalent pipeline (runs locally with pandas/numpy) |
| [tensorflow_fraud_from_spark.py](tensorflow_fraud_from_spark.py) | TensorFlow/Keras equivalent pipeline (runs locally) |
| [credit_card_fraud_detection_big_data_ml.pdf](credit_card_fraud_detection_big_data_ml.pdf) | Full project report |

## Feature Engineering

All four scripts apply the same transformations for a fair comparison:

- **Temporal features**: Hour, day-of-week, and month extracted from the transaction timestamp, then encoded as **sine/cosine pairs** (so 23:00 and 00:00 are treated as adjacent)
- **Age**: Cardholder age computed from date-of-birth relative to the transaction date
- **Distance**: Straight-line distance (km) between the cardholder's home coordinates and the merchant, computed via the **Haversine formula**
- **Categorical encoding**: `gender`, `category`, and `job` label-indexed
- **Standard scaling**: Zero-mean, unit-variance normalization applied to all numeric features
- **PII removal**: Card number, name, address, raw timestamps, and coordinates dropped

## Models

| Framework | Model |
|---|---|
| Spark MLlib | Logistic Regression, Decision Tree (depth 10), Random Forest (100 trees, depth 10) |
| H2O.ai | GLM (logistic), single-tree DRF (depth 10), DRF Random Forest (100 trees, depth 10) |
| TensorFlow | Logistic (1 Dense layer), MLP (128 → 64 → 1, with dropout) |

## Class Imbalance Strategy

All pipelines use **cost-sensitive learning** rather than resampling:

```
w_fraud = n_total / (2 × n_fraud)
w_legit = n_total / (2 × n_legit)
```

The fraud class receives a weight ~86× higher than the legitimate class, preventing the model from simply predicting "not fraud" for everything.

## Running

### Spark (on AWS EMR)

```bash
spark-submit \
  --master yarn --deploy-mode client \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3.11 \
  --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3.11 \
  fraud_detection_spark.py \
  --input s3://csp554-project3/credit_card_transactions.csv \
  --output s3://csp554-project3/output
```

### Spark smoke test (EMR, small sample)

```bash
spark-submit \
  --master yarn --deploy-mode client \
  test_spark.py
```

### H2O (local)

```bash
pip install h2o scikit-learn pandas numpy
python h2o_fraud_from_spark.py \
  --input credit_card_transactions.csv \
  --output ./h2o_output \
  --max_mem 16G
```

### TensorFlow (local)

```bash
pip install tensorflow scikit-learn pandas numpy
python tensorflow_fraud_from_spark.py \
  --input credit_card_transactions.csv \
  --output ./tf_output
```

## Outputs

- **Spark**: model artifacts saved to `OUTPUT/models/{logistic_regression,decision_tree,random_forest}` in S3; metrics summary CSV written to `OUTPUT/metrics/`
- **H2O**: per-model `*_metrics.csv` files and saved H2O model binaries in `OUTPUT/models/`; feature importance CSV for the random forest
- **TensorFlow**: per-model `*_metrics.csv` files and `.keras` model files in `OUTPUT/models/`

## Evaluation Metrics

All pipelines report:

- **AUC-ROC** and **AUC-PR** (PR curve is more informative for heavily imbalanced data)
- **Fraud precision** and **fraud recall** (TP, FP, FN counts)
- Weighted F1, weighted precision, weighted recall (Spark/Keras)

## Dependencies

| Framework | Key Packages |
|---|---|
| Spark | PySpark 3.x (provided by EMR) |
| H2O | `h2o`, `scikit-learn`, `pandas`, `numpy` |
| TensorFlow | `tensorflow`, `scikit-learn`, `pandas`, `numpy` |
