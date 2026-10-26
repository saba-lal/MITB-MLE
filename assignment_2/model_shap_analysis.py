#  Feature contributions (SHAP)
# **Method.** For the live champion, each input's contribution to each loan's score (log-odds of default) is measured against the average loan. 
# For logistic regression this is exact: `coefficient × (input − average input)`
# For XGBoost, the tree-SHAP branch is used 
# # **Importance** = average size (absolute value) of an input's contribution over the loans.
 
 
import glob, os, sys, pickle, json
import numpy as np
import pandas as pd
import pyspark
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
 
spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
 
def read_gold(folder):
    files = sorted(glob.glob(folder + "*.parquet"))
    return spark.read.parquet(*files).toPandas()
 
os.makedirs("charts", exist_ok=True)
INK, MUTED, ACCENT, GRID, PAPER = "#2b2118", "#7a6f64", "#3d405b", "#e8e0d2", "#fbf8f3"
TEAL = "#2a6f6b"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "Palatino", "Palatino Linotype", "Cambria", "DejaVu Serif"],
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": MUTED,
})
 
 
# live champion and its saved preprocessor
champion = json.load(open("model_bank/champion.json"))
with open(f"model_bank/{champion['model_version']}.pkl", "rb") as fh:
    art = pickle.load(fh)
model, prep, cols = art["model"], art["preprocessor"], list(art["feature_cols"])
print("champion:", art["model_version"], "| type:", art["model_type"], "| inputs:", len(cols))
 
# scored loans that have clickstream history (later loans have none, so those inputs would just be filled with medians)
fs = read_gold("datamart/gold/feature_store/")
fs["snapshot_date"] = pd.to_datetime(fs["snapshot_date"])
fs = fs[(fs["snapshot_date"] >= "2023-09-01") & (fs["has_clickstream_history"].astype(bool))].reset_index(drop=True)
print("loans explained:", len(fs), "| months:", fs["snapshot_date"].nunique())
 
# Same clip / fill / scale as scripts/utils/model_preprocessing.apply_preprocessor, using the numbers saved at training.
num = list(prep["numeric_cols"])
X = fs[cols].astype(float).copy()
X[num] = X[num].clip(lower=pd.Series(prep["caps"]["lower"])[num], upper=pd.Series(prep["caps"]["upper"])[num], axis=1)
imp_ = prep["imputer"]
X = X.fillna(pd.Series(imp_.statistics_, index=list(imp_.feature_names_in_)))
sc_ = prep["stdscaler"]
Z = X.copy()
Z[num] = (X[num] - pd.Series(sc_.mean_, index=num)) / pd.Series(sc_.scale_, index=num)
print("saved with:", art["library_versions"], "| running scikit-learn:", __import__("sklearn").__version__)
 
 
# contributions of each input to each loan's score (log-odds), relative to the average loan
if art["model_type"] == "lr":
    coef = pd.Series(model.coef_[0], index=Z.columns)
    S = (Z - Z.mean()) * coef
    base = model.intercept_[0] + (Z.mean() * coef).sum()
    score = model.decision_function(Z)
else:  # xgboost: tree SHAP (not tested in this project, the champion is logistic regression)
    import xgboost as xgb
    booster = model.get_booster()
    contrib = booster.predict(xgb.DMatrix(Z), pred_contribs=True)
    S = pd.DataFrame(contrib[:, :-1], columns=Z.columns, index=Z.index)
    base = contrib[:, -1]
    score = booster.predict(xgb.DMatrix(Z), output_margin=True)
 
# check: contributions + base value reproduce the model's own score
err = np.abs(S.sum(axis=1).to_numpy() + base - score).max()
print("max difference between contributions and model score:", err)
assert err < 1e-6
 
 
imp = S.abs().mean().sort_values(ascending=False)
share = imp / imp.sum()
 
# does a higher input value push the score up (riskier) or down?
sign = {c: np.sign(np.corrcoef(Z[c], S[c])[0, 1]) if Z[c].std() > 0 else 0 for c in imp.index}
 
eng = {"num_fin_products", "emi_to_salary", "debt_to_salary"}
def group(c):
    if c.startswith("fe_"):           return "Clickstream (anonymous)"
    if c in eng:                      return "Engineered ratios"
    if c not in prep["numeric_cols"]: return "Credit and payment flags"
    return "Profile and financial numbers"
 
out = pd.DataFrame({"mean_abs_contribution": imp, "share": share,
                    "higher_value_raises_risk": [sign[c] > 0 for c in imp.index],
                    "group": [group(c) for c in imp.index]})
out["cumulative_share"] = out["share"].cumsum()
out.to_csv("charts/feature_drivers.csv")
 
groups = out.groupby("group")["share"].sum().sort_values(ascending=False)
print(groups.round(3).to_string())
print(f"\ntop 10 inputs: {out['share'].head(10).sum():.1%} | inputs needed for 80%: {int((out['cumulative_share'] < 0.8).sum()) + 1} of {len(out)}")
print(out.head(15).round(3))
 
 
top = out.head(15).iloc[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True)
 
a1.barh(top.index, top["share"] * 100, color=[ACCENT if r else TEAL for r in top["higher_value_raises_risk"]])
a1.set_title("Top 15 inputs by share of total contribution", loc="left", fontsize=11, fontweight="bold")
a1.set_xlabel("% of total average contribution")
a1.legend(handles=[Patch(color=ACCENT, label="higher value raises risk"), Patch(color=TEAL, label="higher value lowers risk")],
          frameon=False, fontsize=8, loc="lower right")
 
g = groups.iloc[::-1]
a2.barh(g.index, g * 100, color=ACCENT)
for y, v in enumerate(g * 100):
    a2.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=9, color=INK)
a2.set_title("By group of inputs", loc="left", fontsize=11, fontweight="bold")
a2.set_xlabel("% of total average contribution")
a2.set_xlim(0, g.max() * 100 * 1.2)
 
for ax in (a1, a2):
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
plt.savefig("charts/feature_drivers.png", dpi=200, bbox_inches="tight")
plt.close()