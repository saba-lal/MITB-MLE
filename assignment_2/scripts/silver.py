import argparse
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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_silver_table

# to call this script: python silver.py --snapshotdate "2023-01-01"

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

    # load arguments
    date_str = snapshotdate
    
    # create silver datalake
    bronze_lms_directory = "datamart/bronze/lms/"
    bronze_attributes_directory = "datamart/bronze/attributes/"
    bronze_financials_directory = "datamart/bronze/financials/"
    bronze_clickstream_directory = "datamart/bronze/clickstream/"

    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    silver_attributes_directory = "datamart/silver/attributes/"
    silver_financials_directory = "datamart/silver/financials/"
    silver_clickstream_directory = "datamart/silver/clickstream/"

    for directory in [silver_loan_daily_directory, silver_attributes_directory,
                    silver_financials_directory, silver_clickstream_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_attributes_directory, silver_attributes_directory, spark)
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_financials_directory, silver_financials_directory, spark)
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)

    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
