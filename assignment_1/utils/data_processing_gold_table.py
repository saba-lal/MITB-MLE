import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def build_gold_feature_store(snapshot_date_str,silver_loan_daily_directory, silver_attributes_directory, silver_financials_directory, silver_clickstream_directory, gold_feature_store_directory, spark, lookback=3):

    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # --- attributes + financials: both already 1 row/customer at month 0 ---
    attr_path = silver_attributes_directory + "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    fin_path = silver_financials_directory + "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'

    df_attr = spark.read.parquet(attr_path)
    df_fin = spark.read.parquet(fin_path)
    df_fin = df_fin.drop("snapshot_date")

    # Attributes and financials silver data is already 1 row per customer
    # For each month, only those records with mob = 0 will be present
    df_features = df_attr.join(df_fin, on="Customer_ID", how="inner")

    # Loan Daily silver data
    # Find loans at mob=0 for current snapshot month
    loan_path = silver_loan_daily_directory + "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    df_loan = (
        spark.read.parquet(loan_path)
        .filter(col("mob") == 0)
        .select("loan_id", "Customer_ID")
    )
    df_features = df_features.join(df_loan, on="Customer_ID", how="inner")

    # Clickstream silver data
    # Lookback window as passed to the function (default 3)
    # aggregate past 3 months of clickstream data for those customers taking loan
    # build the exact prior-month file paths for the lookback window
    candidate_dates = []
    d = snapshot_date - relativedelta(months=1)
    for _ in range(lookback):
        candidate_dates.append(d)
        d = d - relativedelta(months=1)
    
    candidate_files = [
        silver_clickstream_directory + "silver_clickstream_" + d.strftime("%Y_%m_%d") + ".parquet"
        for d in candidate_dates
    ]
    existing_files = [f for f in candidate_files if os.path.exists(f)]
    
    if len(existing_files) == 0: # no files exist
        for c in fe_cols:
            df_features = df_features.withColumn(c, F.lit(None).cast(FloatType()))
        df_features = df_features.withColumn("clickstream_months_used", F.lit(0))
        df_features = df_features.withColumn("has_clickstream_history", F.lit(False))
    else:
        df_click_hist = spark.read.parquet(*existing_files)
    
        agg_exprs = [F.avg(c).cast(FloatType()).alias(c) for c in fe_cols] #if the files exist, average the fe cols by customer_id
        df_click_agg = (
            df_click_hist.groupBy("Customer_ID")
            .agg(*agg_exprs, F.count(F.lit(1)).cast(IntegerType()).alias("clickstream_months_used"))
        )
    
        df_features = df_features.join(df_click_agg, on="Customer_ID", how="left")
        df_features = df_features.fillna({"clickstream_months_used": 0})
        df_features = df_features.withColumn(
            "has_clickstream_history",
            F.when(col("clickstream_months_used") > 0, F.lit(True)).otherwise(F.lit(False))
        )

    # save gold feature store
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df_features.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', df_features.count())

    return df_features