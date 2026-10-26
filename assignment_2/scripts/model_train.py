import argparse
import os
import glob
import pandas as pd
import pickle
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from utils.model_preprocessing import fit_preprocessor, apply_preprocessor

# to call this script: python model_train.py --snapshotdate "2023-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Build a .py script that takes a snapshot date, trains a model and outputs artefact into storage.

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # set up config
    model_train_date_str = snapshotdate
    test_period_months = 1
    val_period_months = 1
    label_lag_months = 6

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["test_period_months"] = test_period_months
    config["val_period_months"] = val_period_months
    config["label_lag_months"] = label_lag_months

    config["latest_labeled_loan_month"] = config["model_train_date"] - relativedelta(months=label_lag_months)
    config["test_start"] = config["latest_labeled_loan_month"] - relativedelta(months=test_period_months - 1)
    config["val_start"]  = config["test_start"] - relativedelta(months=val_period_months)
    config["train_end"]  = config["val_start"] - relativedelta(months=1)

    pprint.pprint(config)

    # connect to label store
    folder_path = "datamart/gold/label_store/"

    def partition_date(f):
        name = os.path.basename(f).replace("gold_label_store_", "").replace(".parquet", "")
        return datetime.strptime(name, "%Y_%m_%d")

    files_list = [f for f in sorted(glob.glob(folder_path + "*.parquet"))
                  if partition_date(f) <= config["model_train_date"]]

    label_store_sdf = spark.read.parquet(*files_list)
    print("partitions used:", len(files_list), "| row_count:", label_store_sdf.count())
    label_store_sdf.show(5)

    folder_path = "datamart/gold/feature_store/"

    def feature_partition_date(f):
        name = os.path.basename(f).replace("gold_feature_store_", "").replace(".parquet", "")
        return datetime.strptime(name, "%Y_%m_%d")

    feature_files = [f for f in sorted(glob.glob(folder_path + "*.parquet"))
                     if feature_partition_date(f) <= config["latest_labeled_loan_month"]]

    features_store_sdf = spark.read.parquet(*feature_files)
    print("partitions used:", len(feature_files), "| row_count:", features_store_sdf.count())
    features_store_sdf.show(5)

    # prepare data for modeling
    data_pdf = (label_store_sdf.select("loan_id", "label")
                .join(features_store_sdf, on="loan_id", how="inner")
                .toPandas())

    print(data_pdf.shape)
    print(data_pdf.groupby("snapshot_date")["label"].agg(["count", "mean"]))
    print(data_pdf.groupby("snapshot_date")["has_clickstream_history"].mean())
    #print(data_pdf["Occupation"].value_counts())

    # split data into train - test
    # TRAIN : labeled loans that started on or before config["train_end"]. Used to fit the candidate models.
    # VAL   : the labeled month(s) from config["val_start"] up to (not including) config["test_start"].
    #         Used to tune hyperparameters AND to choose the model type. After the choice, VAL joins TRAIN
    #         when the tested model is refit.
    # TEST  : the latest labeled month(s). Never used to choose a model. It is scored once for the tested
    #         model, and used for the challenger vs champion check.
    #
    # OOT (out-of-time) = loans that start AFTER this model is frozen. They are scored live by the saved model in the
    # inference task, and compared with the real outcomes once their labels land 6 months later.
    # OOT does not influence training or model selection, so the monitoring chart is unbiased performance measure.
    #
    # Leakage guard: data_pdf contains only loans whose label already exists on the run
    # date (label partitions dated <= model_train_date), so no future labels are used.
    data_pdf["snapshot_date"] = pd.to_datetime(data_pdf["snapshot_date"])

    train_pdf = data_pdf[data_pdf["snapshot_date"] <= config["train_end"]]
    val_pdf   = data_pdf[(data_pdf["snapshot_date"] >= config["val_start"]) & (data_pdf["snapshot_date"] < config["test_start"])]
    test_pdf  = data_pdf[data_pdf["snapshot_date"] >= config["test_start"]]

    assert train_pdf["snapshot_date"].max() < val_pdf["snapshot_date"].min() < test_pdf["snapshot_date"].min(), "sets overlap or are out of order"
    assert len(train_pdf) + len(val_pdf) + len(test_pdf) == len(data_pdf), "rows lost in the split"

    for name, d in [("train", train_pdf), ("val", val_pdf), ("test", test_pdf)]:
        print(f"{name:5s}", len(d), "| months:", sorted(d["snapshot_date"].dt.strftime("%Y-%m").unique()), "| bad rate:", round(d["label"].mean(), 3))

    numeric_cols = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance",
        "num_fin_products", "emi_to_salary", "debt_to_salary",
    ]
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    numeric_cols = numeric_cols + fe_cols
    
    onehot_cols = [c for c in data_pdf.columns if c.startswith(
        ("occupation_", "credit_mix_", "payment_of_min_amount_", "spend_",
         "payment_small", "payment_medium", "payment_large"))]

    feature_cols = numeric_cols + onehot_cols
    print("features:", len(numeric_cols), "numeric +", len(onehot_cols), "one-hot =", len(feature_cols))

    train_pdf[numeric_cols].describe(percentiles=[.01, .05, .5, .95, .99]).T.round(2)

    X_train, y_train = train_pdf[feature_cols].copy(), train_pdf["label"]
    X_val,   y_val   = val_pdf[feature_cols].copy(),   val_pdf["label"]
    X_test,  y_test  = test_pdf[feature_cols].copy(),  test_pdf["label"]

    # preprocessing (clip outliers -> median impute -> scale numeric columns), fitted on TRAIN only.
    # The steps live in utils/model_preprocessing.py so training and inference run the same code.
    prep = fit_preprocessor(X_train, feature_cols, numeric_cols)
    X_train_proc, X_val_proc, X_test_proc = apply_preprocessor(X_train, prep), apply_preprocessor(X_val, prep), apply_preprocessor(X_test, prep)
    for X in (X_train_proc, X_val_proc, X_test_proc):
        assert X.shape[1] == len(feature_cols), "imputer dropped an all-null column"
        assert X.isna().sum().sum() == 0

    print("train:", X_train_proc.shape, "| val:", X_val_proc.shape, "| test:", X_test_proc.shape)
    print("Annual_Income cap:", prep["caps"]["lower"]["Annual_Income"], prep["caps"]["upper"]["Annual_Income"])

    def ks_stat(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.max(tpr - fpr))

    def eval_model(model, cols, label):
        model.fit(X_train_proc[cols], y_train)
        out = []
        for name, X, y in [("train", X_train_proc[cols], y_train), ("val", X_val_proc[cols], y_val), ("test", X_test_proc[cols], y_test)]:
            p = model.predict_proba(X)[:, 1]
            out.append(f"{name} AUC {roc_auc_score(y, p):.3f} KS {ks_stat(y, p):.3f}")
        print(f"{label:24s}", " | ".join(out))

    # Logistic Regression: fit on the training month, score train and test, show the biggest coefficients

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=88).fit(X_train_proc, y_train)

    for name, X, y in [("train", X_train_proc, y_train), ("val", X_val_proc, y_val), ("test", X_test_proc, y_test)]:
        p = lr.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        print(f"{name}: AUC {auc:.3f} | Gini {2*auc-1:.3f} | KS {ks_stat(y, p):.3f}")

    coef = pd.Series(lr.coef_[0], index=X_train_proc.columns).sort_values(key=abs, ascending=False)
    print(coef.head(10).round(3))

    # Compare candidates on the same train/test split, with and without the occupation columns
    no_occ_cols = [c for c in X_train_proc.columns if not c.startswith("occupation_")]
    all_cols = list(X_train_proc.columns)

    xgb_params = dict(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=88, eval_metric="logloss")

    eval_model(LogisticRegression(C=1.0, max_iter=1000, random_state=88), all_cols,    "LR  all 66")
    eval_model(LogisticRegression(C=1.0, max_iter=1000, random_state=88), no_occ_cols, "LR  no occupation (51)")
    eval_model(xgb.XGBClassifier(**xgb_params), all_cols,    "XGB all 66")
    eval_model(xgb.XGBClassifier(**xgb_params), no_occ_cols, "XGB no occupation (51)")

    from itertools import product

    # FEATURE SET DECISION: occupation is excluded from the model (51 features instead of 66:
    # 20 base numeric + 20 clickstream + 11 one-hot).
    # Evidence: in an earlier comparison (before clickstream was added), dropping the 15 occupation columns
    # changed test AUC by only ~0.011 in both LR and XGBoost, within month-to-month noise (~0.03).
    # It leads to a simpler and more explainable model.
    
    model_cols = no_occ_cols

    param_grids = {
        "lr":  {"C": [0.01, 0.1, 1.0, 10.0]},
        "xgb": {"max_depth": [2, 3], "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1], "min_child_weight": [1, 5]},
    }
    base_models = {
        "lr":  lambda **p: LogisticRegression(max_iter=1000, random_state=88, **p),
        "xgb": lambda **p: xgb.XGBClassifier(**{**xgb_params, **p}),
    }

    best_params, val_best = {}, {}
    for name, grid in param_grids.items():
        results = []
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            m = base_models[name](**params).fit(X_train_proc[model_cols], y_train)
            results.append((roc_auc_score(y_val, m.predict_proba(X_val_proc[model_cols])[:, 1]), params))
        val_best[name], best_params[name] = max(results, key=lambda r: r[0])
        print(name, "| best val AUC:", round(val_best[name], 3), "|", best_params[name])

    # Programmatic model selection on VALIDATION: the simplest model within `selection_margin`
    # of the best validation AUC wins. Test is never used to choose.
    selection_margin = 0.05          # AUC points; a more complex model must beat a simpler one by this

    candidates = {name: base_models[name](**best_params[name]) for name in ["lr", "xgb"]}   # simplest -> most complex

    best_val = max(val_best.values())
    chosen = next(name for name in candidates if val_best[name] >= best_val - selection_margin)
    print({k: round(v, 3) for k, v in val_best.items()}, "-> chosen on validation:", chosen)

    # TESTED model: refit on train + validation (preprocessing refit on those rows only), then scored on test.
    # Test is only scored, never used for a choice (apart from the champion promotion check further down).
    import sklearn
    from sklearn.base import clone

    X_tv = pd.concat([X_train[model_cols], X_val[model_cols]])
    y_tv = pd.concat([y_train, y_val])
    num_cols_tv = [c for c in numeric_cols if c in model_cols]
    prep_tv = fit_preprocessor(X_tv, model_cols, num_cols_tv)
    X_tv_proc = apply_preprocessor(X_tv, prep_tv)
    X_test_tv_proc = apply_preprocessor(X_test[model_cols], prep_tv)
    assert X_tv_proc.isna().sum().sum() == 0 and X_test_tv_proc.isna().sum().sum() == 0

    tested = {}
    test_auc = {}
    for name, model in candidates.items():
        tested[name] = clone(model).fit(X_tv_proc[model_cols], y_tv)
        test_auc[name] = roc_auc_score(y_test, tested[name].predict_proba(X_test_tv_proc[model_cols])[:, 1])

    print({k: round(v, 3) for k, v in test_auc.items()}, "<- test AUC, reported only (chosen:", chosen + ")")

    # refit chosen model on all available data (train + val + test) before saving
    import sklearn
    from sklearn.base import clone

    all_pdf = data_pdf
    X_all, y_all = all_pdf[model_cols].copy(), all_pdf["label"]
    num_cols_used = [c for c in numeric_cols if c in model_cols]

    prep_all = fit_preprocessor(X_all, model_cols, num_cols_used)
    X_all_proc = apply_preprocessor(X_all, prep_all)
    assert X_all_proc.isna().sum().sum() == 0

    # --- refit the CHOSEN model type (with its tuned settings) on all labeled months ---
    final_model = clone(candidates[chosen]).fit(X_all_proc[model_cols], y_all)

    # --- scores of the model that was actually TESTED (fitted on train + validation months, before the final refit) ---
    tested_model = tested[chosen]
    train_auc_score = roc_auc_score(y_train, tested_model.predict_proba(X_tv_proc[model_cols].iloc[:len(X_train)])[:, 1])
    val_auc_score   = float(val_best[chosen])   # selection-time: fitted on train only, scored on the validation month
    p_test          = tested_model.predict_proba(X_test_tv_proc[model_cols])[:, 1]
    test_auc_score  = roc_auc_score(y_test, p_test)

    model_artefact = {}
    model_artefact['results'] = {}   # of the model fitted on train + validation, BEFORE the final refit (the refit model has no test score of its own)
    model_artefact['model'] = final_model
    model_artefact['model_version'] = "credit_model_" + config["model_train_date_str"].replace('-', '_')
    model_artefact['model_type'] = chosen
    model_artefact['feature_cols'] = model_cols
    model_artefact['preprocessor'] = prep_all
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_val'] = X_val.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_refit'] = X_all.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(), 2)
    model_artefact['data_stats']['y_val'] = round(y_val.mean(), 2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(), 2)
    model_artefact['data_stats']['y_refit'] = round(y_all.mean(), 2)
    model_artefact['results'] = {}   # of the model BEFORE refit (the refit model has no test score of its own)
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_val'] = val_auc_score
    model_artefact['results']['auc_test'] = test_auc_score
    model_artefact['results']['gini_train'] = round(2*train_auc_score-1, 3)
    model_artefact['results']['gini_val'] = round(2*val_auc_score-1, 3)
    model_artefact['results']['gini_test'] = round(2*test_auc_score-1, 3)
    model_artefact['results']['ks_test'] = round(ks_stat(y_test, p_test), 3)
    model_artefact['hp_params'] = best_params[chosen]
    model_artefact['selection'] = {'margin': selection_margin, 'selected_on': 'validation',
                                   'test_auc_by_candidate': test_auc, 'chosen': chosen,
                                   'val_auc_by_candidate': val_best, 'best_params': best_params}
    model_artefact['library_versions'] = {'sklearn': sklearn.__version__, 'xgboost': xgb.__version__,
                                          'pandas': pd.__version__, 'numpy': np.__version__}

    pprint.pprint({k: v for k, v in model_artefact.items() if k not in ('model', 'preprocessor')})

    # create model_bank dir
    model_bank_directory = "model_bank/"

    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)

    # Full path to the file
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to {file_path}")

    # --- champion vs challenger: promote only if the new model is clearly better ---
    import json
    promotion_margin = 0.02          # challenger test AUC must beat the champion's by this
    champion_path = os.path.join(model_bank_directory, "champion.json")

    champion_auc = None
    if not os.path.exists(champion_path):
        promote = True
        print("No champion yet -> promoting", model_artefact['model_version'])
    else:
        with open(champion_path) as f:
            champion_info = json.load(f)
        with open(os.path.join(model_bank_directory, champion_info["model_version"] + ".pkl"), "rb") as f:
            champion = pickle.load(f)
        # score the current champion on the same newest labeled month (it never saw this month)
        X_champ = apply_preprocessor(test_pdf, champion["preprocessor"])
        champion_auc = float(roc_auc_score(y_test, champion["model"].predict_proba(X_champ)[:, 1]))
        promote = float(test_auc_score) >= champion_auc + promotion_margin
        print("champion", champion_info["model_version"], "test AUC", round(champion_auc, 3),
              "| challenger", model_artefact['model_version'], "test AUC", round(float(test_auc_score), 3),
              "| promote:", promote)

    if promote:
        with open(champion_path, "w") as f:
            json.dump({"model_version": model_artefact['model_version'],
                       "promoted_on": snapshotdate,
                       "challenger_test_auc": round(float(test_auc_score), 4),
                       "previous_champion_test_auc": None if champion_auc is None else round(champion_auc, 4)}, f, indent=2)

    # keep a history of every decision (used for the deck and the SOP)
    with open(os.path.join(model_bank_directory, "promotion_log.jsonl"), "a") as f:
        f.write(json.dumps({"run_date": snapshotdate, "challenger": model_artefact['model_version'],
                            "challenger_test_auc": round(float(test_auc_score), 4),
                            "champion_test_auc": None if champion_auc is None else round(champion_auc, 4),
                            "promoted": bool(promote)}) + "\n")

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)