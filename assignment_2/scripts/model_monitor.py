import argparse
import os
import json
import glob
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pprint
import pyspark

from sklearn.metrics import roc_auc_score, roc_curve

from utils.model_preprocessing import apply_preprocessor

# to call this script: python model_monitor.py --snapshotdate "2024-12-01"

# PSI limits: industry convention (<0.1 stable, 0.1-0.25 watch, >0.25 act)
PSI_WATCH = 0.10
PSI_ACT = 0.25
# performance limits (our own judgement, set from month-to-month AUC noise of ~0.03)
AUC_GAP_WATCH = 0.05
AUC_GAP_ACT = 0.08

MOM_DROP_ALERT = 0.05   # alert if cohort AUC falls more than 5% vs last month's cohort

# no-clickstream share: rise vs the model's training data (based on PSI industry convention thresholds)
NO_CLICK_WATCH = 0.10
NO_CLICK_ACT = 0.25


def psi_status(v):
    if v < PSI_WATCH:
        return "ok"
    if v <= PSI_ACT:
        return "watch"
    return "act"


def psi(ref, cur):
    """Population Stability Index of cur vs ref. Missing values get their own bin."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref_nn = ref[~np.isnan(ref)]
    cur_nn = cur[~np.isnan(cur)]

    values = np.unique(ref_nn)
    if len(values) <= 10:
        # few distinct values (one-hot flags, counts): one bin per value + "other"
        ref_counts = [(ref_nn == v).sum() for v in values]
        cur_counts = [(cur_nn == v).sum() for v in values]
        ref_counts.append(0)
        cur_counts.append(len(cur_nn) - sum(cur_counts))
    else:
        # decile bins cut on the reference data
        edges = np.unique(np.quantile(ref_nn, np.linspace(0, 1, 11)))
        inner = edges[1:-1]
        ref_counts = list(np.bincount(np.searchsorted(inner, ref_nn, side="right"), minlength=len(inner) + 1))
        cur_counts = list(np.bincount(np.searchsorted(inner, cur_nn, side="right"), minlength=len(inner) + 1))

    ref_counts.append(np.isnan(ref).sum())
    cur_counts.append(np.isnan(cur).sum())

    r = np.maximum(np.array(ref_counts, dtype=float) / len(ref), 1e-4)
    c = np.maximum(np.array(cur_counts, dtype=float) / len(cur), 1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_stat(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["model_bank_directory"] = "model_bank/"
    config["label_lag_months"] = 6
    snap_us = snapshotdate.replace('-', '_')

    with open(config["model_bank_directory"] + "champion.json") as file:
        champion_info = json.load(file)
    config["model_version"] = champion_info["model_version"]
    pprint.pprint(config)

    with open(config["model_bank_directory"] + config["model_version"] + ".pkl", 'rb') as file:
        artefact = pickle.load(file)
    model = artefact["model"]
    prep = artefact["preprocessor"]
    model_cols = artefact["feature_cols"]
    train_date = artefact["data_dates"]["model_train_date"]
    latest_labeled = artefact["data_dates"]["latest_labeled_loan_month"]

    def part_date(f, prefix):
        name = os.path.basename(f).replace(prefix, "").replace(".parquet", "")
        return datetime.strptime(name, "%Y_%m_%d")

    # PART 1 - STABILITY: this month's inputs and scores vs the live model's own training data
    # reference = the labeled loans the live model was trained on (same rule as model_train.py)
    label_files = [f for f in sorted(glob.glob("datamart/gold/label_store/*.parquet"))
                   if part_date(f, "gold_label_store_") <= train_date]
    feature_files = [f for f in sorted(glob.glob("datamart/gold/feature_store/*.parquet"))
                     if part_date(f, "gold_feature_store_") <= latest_labeled]
    ref_pdf = (spark.read.parquet(*label_files).select("loan_id", "label")
               .join(spark.read.parquet(*feature_files), on="loan_id", how="inner")
               .toPandas())
    ref_scores = model.predict_proba(apply_preprocessor(ref_pdf, prep))[:, 1]
    print("reference rows:", len(ref_pdf))

    # this month's inputs and scores
    cur_pdf = spark.read.parquet("datamart/gold/feature_store/gold_feature_store_" + snap_us + ".parquet").toPandas()
    pred_pdf = spark.read.parquet("datamart/gold/model_predictions/gold_model_predictions_" + snap_us + ".parquet").toPandas()
    print("current rows:", len(cur_pdf), "| scored rows:", len(pred_pdf))

    run_date = config["snapshot_date"].date()
    rows = []
    feature_psi = {}
    # clickstream inputs are compared only on loans that have clickstream history
    # (loans with no history are zero-filled, which would look like drift)
    ref_hist = ref_pdf["has_clickstream_history"].astype(float) > 0
    cur_hist = cur_pdf["has_clickstream_history"].astype(float) > 0
    for c in model_cols:
        if c.startswith("fe_"):
            if cur_hist.sum() == 0:
                continue   # no loan this month has history: nothing to compare (the share check below catches it)
            feature_psi[c] = psi(ref_pdf.loc[ref_hist, c].astype(float), cur_pdf.loc[cur_hist, c].astype(float))
        else:
            feature_psi[c] = psi(ref_pdf[c].astype(float), cur_pdf[c].astype(float))
        rows.append((run_date, config["model_version"], "psi", c, feature_psi[c], psi_status(feature_psi[c])))

    score_psi = psi(ref_scores, pred_pdf["model_predictions"].values)
    rows.append((run_date, config["model_version"], "psi", "model_score", score_psi, psi_status(score_psi)))

    # share of this month's loans with no clickstream history
    no_click = float(1 - cur_pdf["has_clickstream_history"].astype(float).mean())
    ref_no_click = float(1 - ref_pdf["has_clickstream_history"].astype(float).mean())
    no_click_rise = no_click - ref_no_click
    if no_click_rise > NO_CLICK_ACT:
        no_click_status = "act"
    elif no_click_rise > NO_CLICK_WATCH:
        no_click_status = "watch"
    else:
        no_click_status = "ok"
    rows.append((run_date, config["model_version"], "share", "no_clickstream", no_click, no_click_status))

    # summary: worst of the feature statuses, the score status and no-clickstream status
    statuses = [psi_status(v) for v in feature_psi.values()] + [psi_status(score_psi), no_click_status]
    overall = "act" if "act" in statuses else ("watch" if "watch" in statuses else "ok")
    rows.append((run_date, config["model_version"], "summary", "max_feature_psi", max(feature_psi.values()), overall))
    rows.append((run_date, config["model_version"], "summary", "n_features_watch",
                 float(sum(psi_status(v) == "watch" for v in feature_psi.values())), "n/a"))
    rows.append((run_date, config["model_version"], "summary", "n_features_act",
                 float(sum(psi_status(v) == "act" for v in feature_psi.values())), "n/a"))

    if overall == "act":
        # build the reason from what actually triggered the alert
        causes = []
        act_inputs = [c for c, v in feature_psi.items() if psi_status(v) == "act"]
        if act_inputs:
            causes.append(str(len(act_inputs)) + " input(s) with PSI above " + str(PSI_ACT))
        if psi_status(score_psi) == "act":
            causes.append("score PSI " + str(round(score_psi, 3)) + " above " + str(PSI_ACT))
        if no_click_status == "act":
            causes.append("no-clickstream share rose from " + str(round(ref_no_click, 2)) + " to " + str(round(no_click, 2)))

        alert_dir = "datamart/gold/model_monitoring/alerts/"
        os.makedirs(alert_dir, exist_ok=True)
        with open(alert_dir + "alert_stability_" + snap_us + ".json", "w") as f:
            json.dump({"snapshot_date": snapshotdate, "model_version": config["model_version"],
                       "overall_status": overall, "score_psi": score_psi,
                       "no_clickstream_share": no_click, "reference_no_clickstream_share": ref_no_click,
                       "top_features": [kv for kv in sorted(feature_psi.items(), key=lambda kv: -kv[1])
                                        if psi_status(kv[1]) == "act"][:5],
                       "reason": "; ".join(causes)}, f)
        print("ALERT EVENT (stability) written")
    
    stability_pdf = pd.DataFrame(rows, columns=["snapshot_date", "model_version", "metric", "feature", "value", "status"])
    print(stability_pdf[stability_pdf["metric"] != "psi"])
    print("top 5 PSI:", sorted(feature_psi.items(), key=lambda kv: -kv[1])[:5])

    out_dir = "datamart/gold/model_monitoring/stability/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir + "gold_model_stability_" + snap_us + ".parquet"
    spark.createDataFrame(stability_pdf).write.mode("overwrite").parquet(out_path)
    print("saved to:", out_path, "| rows:", len(stability_pdf))

    # PART 2 - PERFORMANCE: the loans whose labels landed this month (started 6 months ago)
    cohort_month = config["snapshot_date"] - relativedelta(months=config["label_lag_months"])
    cohort_us = cohort_month.strftime("%Y_%m_%d")
    cohort_pred_path = "datamart/gold/model_predictions/gold_model_predictions_" + cohort_us + ".parquet"
    label_path = "datamart/gold/label_store/gold_label_store_" + snap_us + ".parquet"

    if not os.path.exists(cohort_pred_path):
        print("no predictions exist for the", cohort_month.date(), "cohort (before the first model run) -> performance skipped")
    else:
        labels_pdf = spark.read.parquet(label_path).select("loan_id", "label").toPandas()
        cohort_pdf = (spark.read.parquet(cohort_pred_path).toPandas()
                      .merge(labels_pdf, on="loan_id", how="inner"))
        print("cohort", cohort_month.date(), "| labels:", len(labels_pdf), "| scored and labeled:", len(cohort_pdf))

        # the model that actually scored these loans (live at the time), and what it was expected to achieve
        scoring_version = cohort_pdf["model_version"].mode()[0]
        scoring_path = config["model_bank_directory"] + scoring_version + ".pkl"
        expected_auc = np.nan
        if os.path.exists(scoring_path):
            with open(scoring_path, 'rb') as file:
                expected_auc = float(pickle.load(file)["results"]["auc_test"])

        y = cohort_pdf["label"].astype(int).values
        s = cohort_pdf["model_predictions"].values
        auc = float(roc_auc_score(y, s))
        gap = expected_auc - auc

        # month-on-month change vs the previous run's cohort
        mom_drop_pct = np.nan
        prev_auc = np.nan
        prev_mom_path = "datamart/gold/model_monitoring/performance/gold_model_performance_" + \
            (config["snapshot_date"] - relativedelta(months=1)).strftime("%Y_%m_%d") + ".parquet"
        if os.path.exists(prev_mom_path):
            prev_auc = float(spark.read.parquet(prev_mom_path).toPandas()["auc"].iloc[0])
            mom_drop_pct = (prev_auc - auc) / prev_auc



        status = "ok"
        if not np.isnan(gap):
            if gap > AUC_GAP_ACT:
                status = "act"
            elif gap > AUC_GAP_WATCH:
                status = "watch"
                # watch two months in a row -> act
                prev_path = "datamart/gold/model_monitoring/performance/gold_model_performance_" + \
                    (config["snapshot_date"] - relativedelta(months=1)).strftime("%Y_%m_%d") + ".parquet"
                if os.path.exists(prev_path):
                    prev = spark.read.parquet(prev_path).toPandas()
                    if (prev["status"] == "watch").any():
                        status = "act"
                        
        # performance alert: month-on-month drop and/or status "act"
        causes = []
        if (not np.isnan(mom_drop_pct)) and mom_drop_pct > MOM_DROP_ALERT:
            causes.append("cohort AUC fell " + str(round(mom_drop_pct * 100, 1)) + "% vs previous month (limit " + str(MOM_DROP_ALERT * 100) + "%)")
        if status == "act":
            causes.append("performance status act: cohort AUC " + str(round(auc, 3)) + " vs expected " + str(round(expected_auc, 3)))
        alert = len(causes) > 0
        if alert:
            alert_dir = "datamart/gold/model_monitoring/alerts/"
            os.makedirs(alert_dir, exist_ok=True)
            with open(alert_dir + "alert_performance_" + snap_us + ".json", "w") as f:
                json.dump({"snapshot_date": snapshotdate, "cohort_month": str(cohort_month.date()),
                            "auc": auc, "previous_auc": None if np.isnan(prev_auc) else prev_auc,
                            "mom_drop_pct": None if np.isnan(mom_drop_pct) else float(mom_drop_pct),
                            "status": status, "reason": "; ".join(causes)}, f)
            print("ALERT EVENT (performance) written:", "; ".join(causes))

        perf_pdf = pd.DataFrame([{
            "snapshot_date": run_date,
            "cohort_month": cohort_month.date(),
            "scoring_model_version": scoring_version,
            "n": int(len(cohort_pdf)),
            "bad_rate": float(y.mean()),
            "auc": auc,
            "gini": 2 * auc - 1,
            "ks": ks_stat(y, s),
            "expected_auc": expected_auc,
            "auc_gap": gap,
            "status": status,
            "mom_drop_pct": float(mom_drop_pct),
            "alert": alert,
        }])
        print(perf_pdf.T)

        out_dir = "datamart/gold/model_monitoring/performance/"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir + "gold_model_performance_" + snap_us + ".parquet"
        spark.createDataFrame(perf_pdf).write.mode("overwrite").parquet(out_path)
        print("saved to:", out_path)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)