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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
 
    # Process "Age" column
    # extract digits only, discarding any non-numeric characters before or after the digits
    df = df.withColumn("Age", F.regexp_extract(col("Age").cast(StringType()), r"(\d+)", 1).cast(IntegerType()))

    # Ages that are < 18 or > 100 to be nulled
    df = df.withColumn(
        "Age",
        F.when((col("Age") < 18) | (col("Age") > 100), None).otherwise(col("Age"))
    )

    
    # Process "SSN" column
    # "#F%$D@*&8" appears to be a specific repeated value, origin unknown 
    
    # Process "Occupation" column
    # map "________" to "Unknown" category
    df = df.withColumn(
        "Occupation",
        F.when(col("Occupation").rlike("^_+$"), "Unknown").otherwise(col("Occupation"))
    )

    # enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    numeric_int_cols = ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                         "Delay_from_due_date", "Num_of_Delayed_Payment"]
    numeric_float_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit",
                           "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
                           "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]

    for column in numeric_int_cols + numeric_float_cols:
        str_col = col(column).cast(StringType())
        df = df.withColumn(
            column,
            F.when(str_col.rlike(r"^__.*__$"), None)                      # double-underscore wrapped placeholder -> null
             .otherwise(F.regexp_extract(str_col, r"(-?\d+\.?\d*)", 1))   # otherwise: keep the number only
        )

    # enforce schema / data type
    for column in numeric_int_cols:
        df = df.withColumn(column, col(column).cast(IntegerType()))
    for column in numeric_float_cols:
        df = df.withColumn(column, col(column).cast(FloatType()))

    # Process "Credit_History_Age" column
    # "X Years and Y Months" -> decimal years 
    # e.g. 10 Years and 6 Months transform to 10.5
    years = F.regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType())
    months = F.regexp_extract(col("Credit_History_Age"), r"and (\d+) Months", 1).cast(IntegerType())
    df = df.withColumn("Credit_History_Age", (years + months / F.lit(12)).cast(FloatType()))


    # Process "Payment_Behaviour" column
    # map !@9#%8 to an explicit "Unknown" category
    df = df.withColumn(
        "Payment_Behaviour",
        F.when(col("Payment_Behaviour") == "!@9#%8", "Unknown").otherwise(col("Payment_Behaviour"))
    )
    
    # enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Type_of_Loan": StringType(),
        "Credit_Mix": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Payment_Behaviour": StringType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table
    partition_name = "silver_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df

def process_silver_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # enforce schema / data type
    column_type_map = {f"fe_{i}": IntegerType() for i in range(1, 21)}
    column_type_map["Customer_ID"] = StringType()
    column_type_map["snapshot_date"] = DateType()

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df