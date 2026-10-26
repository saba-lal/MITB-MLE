from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_data = BashOperator(
    task_id='dep_check_source_data',
    bash_command=(
        'cd /opt/airflow && '
        'test -f data/lms_loan_daily.csv && '
        'test -f data/features_attributes.csv && '
        'test -f data/features_financials.csv && '
        'test -f data/feature_clickstream.csv'
        ),
    )

    bronze_data_store = BashOperator(
        task_id='bronze_data_store',
        bash_command='cd /opt/airflow && python3 scripts/bronze.py --snapshotdate "{{ ds }}"',
    )
    silver_data_store = BashOperator(
        task_id='silver_data_store',
        bash_command='cd /opt/airflow && python3 scripts/silver.py --snapshotdate "{{ ds }}"',
    )
    gold_data_store = BashOperator(
        task_id='gold_data_store',
        bash_command='cd /opt/airflow && python3 scripts/gold.py --snapshotdate "{{ ds }}"',
    )
    
    label_store_completed = BashOperator(
    task_id='label_store_completed',
    bash_command=(
        "cd /opt/airflow && "
        "test -e datamart/gold/label_store/gold_label_store_{{ ds | replace('-', '_') }}.parquet && "
        "test -e datamart/gold/feature_store/gold_feature_store_{{ ds | replace('-', '_') }}.parquet"
        ),
    )

    # Define task dependencies to run scripts sequentially
    dep_check_source_data >> bronze_data_store >> silver_data_store >> gold_data_store >> label_store_completed

    
    # --- model training ---
    model_train = BashOperator(
        task_id='model_train',
        bash_command='cd /opt/airflow && python3 scripts/model_train.py --snapshotdate "{{ ds }}"',
        start_date=datetime(2023, 9, 1),
    )

    label_store_completed >> model_train

     # --- model inference ---
    model_inference = BashOperator(
        task_id='model_inference',
        bash_command='cd /opt/airflow && python3 scripts/model_inference.py --snapshotdate "{{ ds }}"',
        start_date=datetime(2023, 9, 1),
        trigger_rule='all_done',
    )

    model_train >> model_inference

    # --- model monitoring ---
    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command='cd /opt/airflow && python3 scripts/model_monitor.py --snapshotdate "{{ ds }}"',
        start_date=datetime(2023, 9, 1),
    )

    model_inference >> model_monitor
   
    model_alert = BashOperator(
        task_id='model_alert',
        bash_command=(
            "cd /opt/airflow && "
            "if ls datamart/gold/model_monitoring/alerts/*{{ ds | replace('-', '_') }}.json >/dev/null 2>&1; then "
            "for f in datamart/gold/model_monitoring/alerts/*{{ ds | replace('-', '_') }}.json; do cat $f; echo; done; "
            "echo 'ALERT: placeholder for notifying the monitoring system'; "
            "else echo 'no alert this month'; fi"
        ),
        start_date=datetime(2023, 9, 1),
    )

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed", start_date=datetime(2023, 9, 1))

    model_monitor >> model_alert >> model_monitor_completed