import argparse
import os
import pyspark

import utils.data_processing_gold_table

# to call this script: python gold.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .config("spark.sql.ansi.enabled", "false") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    # silver inputs
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    silver_attributes_directory = "datamart/silver/attributes/"
    silver_financials_directory = "datamart/silver/financials/"
    silver_clickstream_directory = "datamart/silver/clickstream/"

    # gold outputs
    gold_label_store_directory = "datamart/gold/label_store/"
    gold_feature_store_directory = "datamart/gold/feature_store/"

    for directory in [gold_label_store_directory, gold_feature_store_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # label store: default = 30 days past due at month on book 6
    utils.data_processing_gold_table.process_labels_gold_table(
        date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6)

    # feature store
    utils.data_processing_gold_table.build_gold_feature_store(
        date_str, silver_loan_daily_directory, silver_attributes_directory, silver_financials_directory,
        silver_clickstream_directory, gold_feature_store_directory, spark)

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)