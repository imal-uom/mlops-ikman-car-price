"""Compare a linear model with the EBM (main effects only, and with interactions) on the same
data, all tracked in MLflow.

This is exploration, separate from the production pipeline in dvc.yaml.
"""

import mlflow
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import tracking
from src.config import ROOT, load_params
from src.features import (
    CAT_FEATURES,
    NUM_FEATURES,
    TARGET,
    apply_spec,
    build_features,
    fit_spec,
)
from src.metrics import score


def main() -> None:
    tracking.configure()
    params = load_params()
    data_dir = ROOT / params["data"]["processed_dir"]
    train = build_features(pd.read_csv(data_dir / "train.csv"))
    val = build_features(pd.read_csv(data_dir / "eval.csv"))
    y_tr, y_val = np.log(train[TARGET]), val[TARGET]

    # Ridge on one-hot categoricals: the classic hedonic price model.
    ridge = make_pipeline(
        ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CAT_FEATURES),
                (
                    "num",
                    Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                    NUM_FEATURES,
                ),
            ]
        ),
        RidgeCV(alphas=np.logspace(-2, 2, 20)),
    )
    spec = fit_spec(train, prior=params["model"]["typical_prior"])
    X_tr, X_val = apply_spec(train, spec), apply_spec(val, spec)
    ebm = params["model"]["ebm"]
    models = {
        "ridge": (ridge, train[CAT_FEATURES + NUM_FEATURES], val[CAT_FEATURES + NUM_FEATURES]),
        "ebm-main-effects": (
            ExplainableBoostingRegressor(**ebm | {"interactions": 0}),
            X_tr,
            X_val,
        ),
        "ebm-interactions": (ExplainableBoostingRegressor(**ebm), X_tr, X_val),
    }

    mlflow.set_experiment(params["model"]["experiment"] + "-experiments")
    results = []
    for name, (model, X_fit, X_eval) in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_fit, y_tr)
            metrics = score(y_val, model.predict(X_eval))
            mlflow.log_param("model_family", name)
            mlflow.log_metrics(metrics)
        results.append({"model": name, **metrics})

    table = pd.DataFrame(results).set_index("model").round(4)
    print(table[["median_ape", "mape", "within_20", "rmse_log"]].sort_values("median_ape"))


if __name__ == "__main__":
    main()
