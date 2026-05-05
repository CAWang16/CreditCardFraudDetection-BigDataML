import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

# run with spark-submit, not plain python
# spark-submit --master yarn --deploy-mode client \
#   --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3.11 \
#   --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3.11 \
#   test_spark.py

S3_INPUT = "s3://csp554-project3/credit_card_transactions.csv"
SAMPLE_N = 10_000

spark = SparkSession.builder.appName("FraudDetection-Test").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# pull a small sample from S3 to keep the test fast
print("\n[1] Reading from S3...")
raw = spark.read.option("header", "true").option("inferSchema", "true").csv(S3_INPUT)
total = raw.count()
df = raw.sample(fraction=min(SAMPLE_N / total, 1.0), seed=42)
actual_n = df.count()
fraud_n = df.filter(F.col("is_fraud") == 1).count()
print(f"    Total rows in S3 file : {total:,}")
print(f"    Sample size           : {actual_n:,}")
print(f"    Fraud in sample       : {fraud_n} ({fraud_n/actual_n*100:.2f}%)")
print(f"    Schema: {df.columns}")

print("\n[2] Engineering features...")

df = (
    df.withColumn("trans_ts", F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"))
      .withColumn("dob_ts", F.to_date("dob", "yyyy-MM-dd"))
)

df = (
    df.withColumn("hour",        F.hour("trans_ts"))
      .withColumn("day_of_week", F.dayofweek("trans_ts") - 1)
      .withColumn("month",       F.month("trans_ts"))
      .withColumn("age",        (F.datediff(F.col("trans_ts"), F.col("dob_ts")) / 365).cast("int"))
)

# straight-line distance between cardholder home and merchant in km
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

# encode hour/day/month as sin+cos so the model knows 23:00 and 00:00 are close
df = (
    df.withColumn("hour_sin",  F.sin(2 * math.pi * F.col("hour") / 24))
      .withColumn("hour_cos",  F.cos(2 * math.pi * F.col("hour") / 24))
      .withColumn("dow_sin",   F.sin(2 * math.pi * F.col("day_of_week") / 7))
      .withColumn("dow_cos",   F.cos(2 * math.pi * F.col("day_of_week") / 7))
      .withColumn("month_sin", F.sin(2 * math.pi * (F.col("month") - 1) / 12))
      .withColumn("month_cos", F.cos(2 * math.pi * (F.col("month") - 1) / 12))
)

DROP_COLS = [
    "trans_date_trans_time", "dob", "trans_ts", "dob_ts",
    "cc_num", "trans_num", "first", "last", "street",
    "city", "state", "zip", "lat", "long",
    "merch_lat", "merch_long", "unix_time",
    "merchant", "merch_zipcode", "_c0", "Unnamed: 0",
    "hour", "day_of_week", "month",
]
df = df.drop(*[c for c in DROP_COLS if c in df.columns])
df = df.withColumnRenamed("is_fraud", "label").withColumn("label", F.col("label").cast("double"))
print(f"    Final columns: {df.columns}")

# the dataset is heavily imbalanced (~0.58% fraud), so we assign higher
# loss weight to fraud transactions instead of resampling
n_total = df.count()
n_fraud = df.filter(F.col("label") == 1).count()
n_legit = n_total - n_fraud
w_fraud = n_total / (2.0 * max(n_fraud, 1))
w_legit = n_total / (2.0 * max(n_legit, 1))
df = df.withColumn("classWeight", F.when(F.col("label") == 1, w_fraud).otherwise(w_legit))
print(f"\n[3] Class weights — fraud: {w_fraud:.2f}, legit: {w_legit:.4f}")

train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"\n[4] Train: {train.count():,}  |  Test: {test.count():,}")

CAT_COLS     = ["gender", "category", "job"]
NUMERIC_COLS = ["amt", "city_pop", "age", "distance_km",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos"]
IDX_COLS     = [f"{c}_idx" for c in CAT_COLS]
ALL_FEATURES = NUMERIC_COLS + IDX_COLS

indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CAT_COLS]
assembler = VectorAssembler(inputCols=ALL_FEATURES, outputCol="raw_features", handleInvalid="keep")
scaler    = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
shared    = indexers + [assembler, scaler]

auc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
f1_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

def run_model(name, classifier):
    print(f"\n[Model] {name}")
    pipeline = Pipeline(stages=shared + [classifier])
    model    = pipeline.fit(train)
    preds    = model.transform(test)
    auc  = auc_eval.evaluate(preds)
    f1   = f1_eval.evaluate(preds)
    tp   = preds.filter((F.col("label")==1) & (F.col("prediction")==1)).count()
    fp   = preds.filter((F.col("label")==0) & (F.col("prediction")==1)).count()
    fn   = preds.filter((F.col("label")==1) & (F.col("prediction")==0)).count()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"    AUC-ROC        : {auc:.4f}")
    print(f"    F1 (weighted)  : {f1:.4f}")
    print(f"    Fraud Precision: {prec:.4f}  (TP={tp}, FP={fp})")
    print(f"    Fraud Recall   : {rec:.4f}  (TP={tp}, FN={fn})")
    return model

print("\n" + "="*50)
print("TRAINING MODELS ON SAMPLE DATA")
print("="*50)

lr_model = run_model(
    "Logistic Regression",
    LogisticRegression(
        featuresCol="features", labelCol="label", weightCol="classWeight",
        maxIter=50, regParam=0.01, family="binomial"
    )
)

dt_model = run_model(
    "Decision Tree",
    DecisionTreeClassifier(
        featuresCol="features", labelCol="label", weightCol="classWeight",
        maxDepth=8, maxBins=32, seed=42
    )
)

rf_model = run_model(
    "Random Forest",
    RandomForestClassifier(
        featuresCol="features", labelCol="label", weightCol="classWeight",
        numTrees=20, maxDepth=8, maxBins=32, seed=42
    )
)

rf_stage = rf_model.stages[-1]
importances = sorted(zip(ALL_FEATURES, rf_stage.featureImportances.toArray()),
                     key=lambda x: x[1], reverse=True)
print("\n  Top 5 Feature Importances (Random Forest):")
for feat, imp in importances[:5]:
    print(f"    {feat:<20s} {imp:.4f}")

print("\n" + "="*50)
print("TEST COMPLETE — All 3 models ran successfully on EMR + S3")
print("="*50)
spark.stop()
