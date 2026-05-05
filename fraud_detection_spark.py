import argparse
import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

# run with:
# spark-submit --master yarn --deploy-mode client \
#   --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3.11 \
#   --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3.11 \
#   fraud_detection_spark.py \
#   --input s3://csp554-project3/credit_card_transactions.csv \
#   --output s3://csp554-project3/output

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True)
parser.add_argument("--output", required=True)
args, _ = parser.parse_known_args()

INPUT_PATH  = args.input
OUTPUT_PATH = args.output.rstrip("/")

spark = (
    SparkSession.builder
    .appName("FraudDetection")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"\n[1] Loading data from {INPUT_PATH}")
raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(INPUT_PATH)
)
print(f"    Rows: {raw.count():,}")

print("\n[2] Engineering features...")

df = raw.withColumn(
    "trans_ts", F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss")
).withColumn(
    "dob_ts", F.to_date("dob", "yyyy-MM-dd")
)

df = (
    df.withColumn("hour",        F.hour("trans_ts"))
      .withColumn("day_of_week", F.dayofweek("trans_ts") - 1)
      .withColumn("month",       F.month("trans_ts"))
)

df = df.withColumn(
    "age",
    (F.datediff(F.col("trans_ts"), F.col("dob_ts")) / 365).cast("int")
)

# straight-line distance between cardholder home and the merchant in km
@F.udf(returnType=DoubleType())
def haversine_udf(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))

df = df.withColumn("distance_km", haversine_udf("lat", "long", "merch_lat", "merch_long"))

# drop PII and columns we've already decomposed into features
DROP_COLS = [
    "trans_date_trans_time", "dob", "trans_ts", "dob_ts",
    "cc_num", "trans_num", "first", "last", "street",
    "city", "state", "zip", "lat", "long",
    "merch_lat", "merch_long", "unix_time",
    "merchant", "merch_zipcode", "_c0", "Unnamed: 0",
]
df = df.drop(*[c for c in DROP_COLS if c in df.columns])
df = df.withColumnRenamed("is_fraud", "label")

# encode hour/day/month as sin+cos pairs so the model treats them as circular
# e.g. 23:00 and 00:00 should be close together, not far apart
df = (
    df.withColumn("hour_sin",  F.sin(2 * math.pi * F.col("hour") / 24))
      .withColumn("hour_cos",  F.cos(2 * math.pi * F.col("hour") / 24))
      .withColumn("dow_sin",   F.sin(2 * math.pi * F.col("day_of_week") / 7))
      .withColumn("dow_cos",   F.cos(2 * math.pi * F.col("day_of_week") / 7))
      .withColumn("month_sin", F.sin(2 * math.pi * (F.col("month") - 1) / 12))
      .withColumn("month_cos", F.cos(2 * math.pi * (F.col("month") - 1) / 12))
      .drop("hour", "day_of_week", "month")
)

print(f"    Columns after engineering: {df.columns}")

print("\n[3] Encoding categorical features...")
CAT_COLS = ["gender", "category", "job"]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in CAT_COLS
]

NUMERIC_COLS = ["amt", "city_pop", "age", "distance_km",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos"]
IDX_COLS     = [f"{c}_idx" for c in CAT_COLS]
ALL_FEATURES = NUMERIC_COLS + IDX_COLS

# split separately on fraud/legit rows to keep the fraud ratio consistent
# in both train and test sets
print("\n[4] Splitting 80/20 (stratified by label)...")
fraud     = df.filter(F.col("label") == 1)
legit     = df.filter(F.col("label") == 0)
train_f, test_f = fraud.randomSplit([0.8, 0.2], seed=42)
train_l, test_l = legit.randomSplit([0.8, 0.2], seed=42)
train = train_f.union(train_l)
test  = test_f.union(test_l)
print(f"    Train: {train.count():,}  |  Test: {test.count():,}")

# only 0.58% of transactions are fraud, so we weight the minority class higher
# to stop the model from just predicting "not fraud" for everything
print("\n[5] Computing class weights for imbalance...")
n_total = train.count()
n_fraud = train.filter(F.col("label") == 1).count()
n_legit = n_total - n_fraud
w_fraud = n_total / (2.0 * n_fraud)
w_legit = n_total / (2.0 * n_legit)

train = train.withColumn(
    "classWeight",
    F.when(F.col("label") == 1, w_fraud).otherwise(w_legit)
)
test = test.withColumn("classWeight", F.lit(1.0))
print(f"    Fraud weight: {w_fraud:.2f}  |  Legit weight: {w_legit:.4f}")

assembler = VectorAssembler(inputCols=ALL_FEATURES, outputCol="raw_features", handleInvalid="keep")
scaler    = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
shared_stages = indexers + [assembler, scaler]

auc_eval  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
pr_eval   = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
f1_eval   = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

def evaluate(predictions, model_name):
    auc  = auc_eval.evaluate(predictions)
    pr   = pr_eval.evaluate(predictions)
    f1   = f1_eval.evaluate(predictions)
    prec = prec_eval.evaluate(predictions)
    rec  = rec_eval.evaluate(predictions)

    tp = predictions.filter((F.col("label")==1) & (F.col("prediction")==1)).count()
    fp = predictions.filter((F.col("label")==0) & (F.col("prediction")==1)).count()
    fn = predictions.filter((F.col("label")==1) & (F.col("prediction")==0)).count()
    fraud_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    fraud_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n  -- {model_name} --")
    print(f"    AUC-ROC        : {auc:.4f}")
    print(f"    AUC-PR         : {pr:.4f}")
    print(f"    F1 (weighted)  : {f1:.4f}")
    print(f"    Precision (w)  : {prec:.4f}")
    print(f"    Recall (w)     : {rec:.4f}")
    print(f"    Fraud Precision: {fraud_prec:.4f}  ({tp} TP / {tp+fp} predicted fraud)")
    print(f"    Fraud Recall   : {fraud_rec:.4f}  ({tp} TP / {tp+fn} actual fraud)")

    return {
        "model": model_name, "auc_roc": auc, "auc_pr": pr,
        "f1": f1, "precision": prec, "recall": rec,
        "fraud_precision": fraud_prec, "fraud_recall": fraud_rec,
    }

print("\n[6] Training Logistic Regression...")
lr = LogisticRegression(
    featuresCol="features", labelCol="label", weightCol="classWeight",
    maxIter=100, regParam=0.01, elasticNetParam=0.0, family="binomial",
)
lr_pipeline = Pipeline(stages=shared_stages + [lr])
lr_model    = lr_pipeline.fit(train)
lr_preds    = lr_model.transform(test)
lr_metrics  = evaluate(lr_preds, "Logistic Regression")
lr_model.save(f"{OUTPUT_PATH}/models/logistic_regression")

print("\n[7] Training Decision Tree...")
dt = DecisionTreeClassifier(
    featuresCol="features", labelCol="label", weightCol="classWeight",
    maxDepth=10, maxBins=64, seed=42,
)
dt_pipeline = Pipeline(stages=shared_stages + [dt])
dt_model    = dt_pipeline.fit(train)
dt_preds    = dt_model.transform(test)
dt_metrics  = evaluate(dt_preds, "Decision Tree")
dt_model.save(f"{OUTPUT_PATH}/models/decision_tree")

print("\n[8] Training Random Forest...")
rf = RandomForestClassifier(
    featuresCol="features", labelCol="label", weightCol="classWeight",
    numTrees=100, maxDepth=10, maxBins=64,
    featureSubsetStrategy="sqrt",
    seed=42,
)
rf_pipeline = Pipeline(stages=shared_stages + [rf])
rf_model    = rf_pipeline.fit(train)
rf_preds    = rf_model.transform(test)
rf_metrics  = evaluate(rf_preds, "Random Forest")
rf_model.save(f"{OUTPUT_PATH}/models/random_forest")

rf_stage = rf_model.stages[-1]
importances = sorted(
    zip(ALL_FEATURES, rf_stage.featureImportances.toArray()),
    key=lambda x: x[1], reverse=True
)
print("\n  Random Forest — Feature Importances:")
for feat, imp in importances:
    print(f"    {feat:<20s} {imp:.4f}")

print("\n[9] Saving metrics summary...")
metrics_rows = [lr_metrics, dt_metrics, rf_metrics]
metrics_df   = spark.createDataFrame(metrics_rows)
(
    metrics_df.coalesce(1)
    .write.mode("overwrite")
    .option("header", "true")
    .csv(f"{OUTPUT_PATH}/metrics")
)
print(f"    Written to {OUTPUT_PATH}/metrics/")

print("\nDone.")
spark.stop()
