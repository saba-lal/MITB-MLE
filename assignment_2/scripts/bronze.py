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

import utils.data_processing_bronze_table

# to call this script: python bronze.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create bronze datalake
    bronze_lms_directory = "datamart/bronze/lms/"
    bronze_attributes_directory = "datamart/bronze/attributes/"
    bronze_financials_directory = "datamart/bronze/financials/"
    bronze_clickstream_directory = "datamart/bronze/clickstream/"

    for directory in [bronze_lms_directory, bronze_attributes_directory, bronze_financials_directory, bronze_clickstream_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # run data processing
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/lms_loan_daily.csv", bronze_lms_directory, "loan_daily", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/features_attributes.csv", bronze_attributes_directory, "attributes", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/features_financials.csv", bronze_financials_directory, "financials", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/feature_clickstream.csv", bronze_clickstream_directory, "clickstream", spark)
    
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
