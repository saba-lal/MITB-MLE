import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_preprocessor(X_train, feature_cols, numeric_cols):
    """Learn caps, medians and scaling from the training rows only."""
    X = X_train[feature_cols].copy()
    
    caps = {
        "lower": X[numeric_cols].quantile(0.01),
        "upper": X[numeric_cols].quantile(0.99),
    }
    X[numeric_cols] = X[numeric_cols].clip(lower=caps["lower"], upper=caps["upper"], axis=1)

    imputer = SimpleImputer(strategy="median").set_output(transform="pandas").fit(X)
    X_imp = imputer.transform(X)

    stdscaler = StandardScaler().fit(X_imp[numeric_cols])

    return {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "caps": caps,
        "imputer": imputer,
        "stdscaler": stdscaler,
    }


def apply_preprocessor(X, prep):
    """Apply a saved preprocessor to any rows (training, test or new loans)."""
    X = X[prep["feature_cols"]].copy()
    num = prep["numeric_cols"]

    X[num] = X[num].clip(lower=prep["caps"]["lower"], upper=prep["caps"]["upper"], axis=1)
    X_imp = prep["imputer"].transform(X)
    X_imp[num] = prep["stdscaler"].transform(X_imp[num])
    return X_imp