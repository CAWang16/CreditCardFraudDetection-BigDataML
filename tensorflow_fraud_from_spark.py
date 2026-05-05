import argparse
import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
FEATURES = [
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


def compute_class_weights(y_train):
    n_total = len(y_train)
    n_fraud = int(y_train.sum())
    n_legit = n_total - n_fraud
    w_fraud = n_total / (2.0 * n_fraud)
    w_legit = n_total / (2.0 * n_legit)
    return {0: w_legit, 1: w_fraud}, w_fraud, w_legit


def build_logistic_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_mlp_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def evaluate_model(model, x_test, y_test, output_dir, model_name):
    y_prob = model.predict(x_test, batch_size=8192, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob),
        "fraud_precision": precision_score(y_test, y_pred, zero_division=0),
        "fraud_recall": recall_score(y_test, y_pred, zero_division=0),
        "tp": int(((y_test == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_test == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_test == 1) & (y_pred == 0)).sum()),
    }
    print(f"\n-- {model_name} --")
    for k in ["auc_roc", "auc_pr", "fraud_precision", "fraud_recall"]:
        print(f"{k}: {metrics[k]:.4f}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
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

    scaler = StandardScaler(with_mean=True, with_std=True)
    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    test_df[FEATURES] = scaler.transform(test_df[FEATURES])

    x_train = train_df[FEATURES].astype("float32").to_numpy()
    y_train = train_df["label"].astype("float32").to_numpy()
    x_test = test_df[FEATURES].astype("float32").to_numpy()
    y_test = test_df["label"].astype("float32").to_numpy()

    class_weight, w_fraud, w_legit = compute_class_weights(y_train)
    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    print(f"Fraud weight: {w_fraud:.2f}  Legit weight: {w_legit:.4f}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=3, restore_best_weights=True)
    ]

    print("[3] Training TensorFlow logistic-style model...")
    logistic = build_logistic_model(x_train.shape[1])
    logistic.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=4096,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    evaluate_model(logistic, x_test, y_test, args.output, "tf_logistic")
    logistic.save(os.path.join(args.output, "models", "tf_logistic.keras"))

    print("[4] Training TensorFlow MLP model...")
    mlp = build_mlp_model(x_train.shape[1])
    mlp.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=4096,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    evaluate_model(mlp, x_test, y_test, args.output, "tf_mlp")
    mlp.save(os.path.join(args.output, "models", "tf_mlp.keras"))


if __name__ == "__main__":
    main()
