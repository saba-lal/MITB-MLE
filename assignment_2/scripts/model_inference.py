import argparse
import os
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import pprint
import pyspark

from utils.model_preprocessing import apply_preprocessor

# to call this script: python model_inference.py --snapshotdate "2023-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_bank_directory"] = "model_bank/"

    # the live model is whichever one the champion pointer names
    with open(config["model_bank_directory"] + "champion.json") as file:
        champion_info = json.load(file)
    config["model_version"] = champion_info["model_version"]
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_version"] + ".pkl"

    pprint.pprint(config)

    # --- load model artefact from model bank ---
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)

    print("Model loaded successfully! " + config["model_artefact_filepath"])

    # --- load feature store (this month's partition) ---
    feature_location = "datamart/gold/feature_store/gold_feature_store_" + config["snapshot_date_str"].replace('-', '_') + ".parquet"
    features_sdf = spark.read.parquet(feature_location)
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date"])

    features_pdf = features_sdf.toPandas()

    # --- preprocess data for modeling ---
    # same clip -> impute -> scale steps as training, using the numbers saved in the artefact
    X_inference = apply_preprocessor(features_pdf, model_artefact["preprocessor"])

    print('X_inference', X_inference.shape[0])

    # --- model prediction inference ---
    model = model_artefact["model"]
    y_inference = model.predict_proba(X_inference)[:, 1]

    # prepare output
    y_inference_pdf = features_pdf[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_version"] = config["model_version"]
    y_inference_pdf["model_predictions"] = y_inference

    assert len(y_inference_pdf) == len(features_pdf), "rows lost"
    assert y_inference_pdf["model_predictions"].notna().all(), "missing scores"
    assert y_inference_pdf["loan_id"].is_unique, "duplicate loan_id"

    # --- save model inference to datamart gold table ---
    gold_directory = "datamart/gold/model_predictions/"
    print(gold_directory)

    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    partition_name = "gold_model_predictions_" + config["snapshot_date_str"].replace('-', '_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, '| rows:', len(y_inference_pdf))

    # --- end spark session ---
    spark.stop()

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)