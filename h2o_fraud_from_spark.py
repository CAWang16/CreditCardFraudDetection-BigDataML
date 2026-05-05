import argparse
import math
import os
import numpy as np
import pandas as pd
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DROP_COLS = [
    "trans_date_trans_time", "dob", "trans_ts", "dob_ts",
    "cc_num", "trans_num", "first", "last", "street",
    "city", "state", "zip", "lat", "long",
    "merch_lat", "merch_long", "unix_time",
    "merchant", "merch_zipcode", "_c0", "Unnamed: 0",
]

CATEGORICAL_COLS = ["gender", "category", "job"]
BASE_FEATURES = [
    "amt", "city_pop", "age", "distance_km",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "gender_idx", "category_idx", "job_idx",
]


def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(a))


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_ts"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["dob_ts"] = pd.to_datetime(df["dob"], errors="coerce")

    df["hour"] = df["trans_ts"].dt.hour
    df["day_of_week"] = df["trans_ts"].dt.dayofweek
    df["month"] = df["trans_ts"].dt.month
    df["age"] = ((df["trans_ts"] - df["dob_ts"]).dt.days / 365.0).fillna(0).astype(int)

    df["distance_km"] = haversine_np(
        df["lat"].astype(float),
        df["long"].astype(float),
        df["merch_lat"].astype(float),
        df["merch_long"].astype(float),
    )

    df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * math.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * math.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * math.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * math.pi * (df["month"] - 1) / 12.0)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in ["hour", "day_of_week", "month"] if c in df.columns], errors="ignore")
    df = df.rename(columns={"is_fraud": "label"})
    df["label"] = df["label"].astype(int)
    return df


def build_index_mappings(train_df: pd.DataFrame):
    mappings = {}
    for col in CATEGORICAL_COLS:
        values = sorted(train_df[col].fillna("__MISSING__").astype(str).unique().tolist())
        mappings[col] = {v: i for i, v in enumerate(values)}
    return mappings


def apply_index_mappings(df: pd.DataFrame, mappings):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        key = df[col].fillna("__MISSING__").astype(str)
        df[f"{col}_idx"] = key.map(mappings[col]).fillna(-1).astype(int)
    return df.drop(columns=CATEGORICAL_COLS)


def add_class_weights(train_df: pd.DataFrame, test_df: pd.DataFrame):
    n_total = len(train_df)
    n_fraud = int(train_df["label"].sum())
    n_legit = n_total - n_fraud
    w_fraud = n_total / (2.0 * n_fraud)
    w_legit = n_total / (2.0 * n_legit)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["classWeight"] = np.where(train_df["label"] == 1, w_fraud, w_legit)
    test_df["classWeight"] = 1.0
    return train_df, test_df, w_fraud, w_legit


def scale_like_spark(train_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[BASE_FEATURES] = scaler.fit_transform(train_df[BASE_FEATURES])
    test_df[BASE_FEATURES] = scaler.transform(test_df[BASE_FEATURES])
    return train_df, test_df


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "fraud_precision": precision_score(y_true, y_pred, zero_division=0),
        "fraud_recall": recall_score(y_true, y_pred, zero_division=0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_h2o_model(model, test_hf, output_dir, model_name):
    pred_df = h2o.as_list(model.predict(test_hf), use_pandas=True)
    y_true = h2o.as_list(test_hf["label"], use_pandas=True).iloc[:, 0].astype(int).to_numpy()
    y_prob = pred_df["p1"].astype(float).to_numpy()
    metrics = compute_metrics(y_true, y_prob)
    print(f"\n-- {model_name} --")
    for k in ["auc_roc", "auc_pr", "fraud_precision", "fraud_recall"]:
        print(f"{k}: {metrics[k]:.4f}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local path to credit_card_transactions.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_mem", default="16G")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "models"), exist_ok=True)

    print("[1] Reading CSV...")
    df = pd.read_csv(args.input)
    print(f"Rows: {len(df):,}")

    print("[2] Preprocessing exactly like Spark...")
    df = preprocess(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    mappings = build_index_mappings(train_df)
    train_df = apply_index_mappings(train_df, mappings)
    test_df = apply_index_mappings(test_df, mappings)

    train_df, test_df, w_fraud, w_legit = add_class_weights(train_df, test_df)
    train_df, test_df = scale_like_spark(train_df, test_df)

    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    print(f"Fraud weight: {w_fraud:.2f}  Legit weight: {w_legit:.4f}")

    print("[3] Starting H2O...")
    h2o.init(max_mem_size=args.max_mem)

    train_hf = h2o.H2OFrame(train_df)
    test_hf = h2o.H2OFrame(test_df)

    x = BASE_FEATURES
    y = "label"
    weight_col = "classWeight"

    print("[4] Training H2O GLM (logistic regression)...")
    glm = H2OGeneralizedLinearEstimator(
        family="binomial",
        alpha=0.0,
        lambda_=0.01,
        weights_column=weight_col,
        seed=42,
    )
    glm.train(x=x, y=y, training_frame=train_hf)
    evaluate_h2o_model(glm, test_hf, args.output, "h2o_glm")
    h2o.save_model(glm, path=os.path.join(args.output, "models"), force=True)

    print("[5] Training H2O DRF with 1 tree (closest H2O stand-in for single decision tree)...")
    cart_like = H2ORandomForestEstimator(
        ntrees=1,
        max_depth=10,
        sample_rate=1.0,
        mtries=len(x),
        weights_column=weight_col,
        seed=42,
    )
    cart_like.train(x=x, y=y, training_frame=train_hf)
    evaluate_h2o_model(cart_like, test_hf, args.output, "h2o_single_tree_like")
    h2o.save_model(cart_like, path=os.path.join(args.output, "models"), force=True)

    print("[6] Training H2O DRF random forest...")
    rf = H2ORandomForestEstimator(
        ntrees=100,
        max_depth=10,
        weights_column=weight_col,
        seed=42,
    )
    rf.train(x=x, y=y, training_frame=train_hf)
    evaluate_h2o_model(rf, test_hf, args.output, "h2o_random_forest")
    h2o.save_model(rf, path=os.path.join(args.output, "models"), force=True)

    varimp = rf.varimp(use_pandas=True)
    if varimp is not None:
        varimp.to_csv(os.path.join(args.output, "h2o_random_forest_varimp.csv"), index=False)
        print(varimp.head(10))

    h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()
