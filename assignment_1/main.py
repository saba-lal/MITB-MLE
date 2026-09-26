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
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"
bronze_attributes_directory = "datamart/bronze/attributes/"
bronze_financials_directory = "datamart/bronze/financials/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"

for directory in [bronze_lms_directory, bronze_attributes_directory, bronze_financials_directory, bronze_clickstream_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/lms_loan_daily.csv", bronze_lms_directory, "loan_daily", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/features_attributes.csv", bronze_attributes_directory, "attributes", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/features_financials.csv", bronze_financials_directory, "financials", spark)
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/feature_clickstream.csv", bronze_clickstream_directory, "clickstream", spark)


# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"
silver_clickstream_directory = "datamart/silver/clickstream/"

for directory in [silver_loan_daily_directory, silver_attributes_directory, silver_financials_directory, silver_clickstream_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    utils.data_processing_silver_table.process_silver_attributes_table(date_str, bronze_attributes_directory, silver_attributes_directory, spark)
    utils.data_processing_silver_table.process_silver_financials_table(date_str, bronze_financials_directory, silver_financials_directory, spark)
    utils.data_processing_silver_table.process_silver_clickstream_table(date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)


# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"
for directory in [gold_label_store_directory, gold_feature_store_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# run gold backfill - label store and feature store
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
    utils.data_processing_gold_table.build_gold_feature_store(date_str, silver_loan_daily_directory, silver_attributes_directory, silver_financials_directory, 
                                                              silver_clickstream_directory, gold_feature_store_directory, spark )
    

folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df_goldfeatures = spark.read.option("header", "true").parquet(*files_list)
print("row_count:", df_goldfeatures.count())

df_goldfeatures.printSchema()


    