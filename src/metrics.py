"""Model quality metrics shared by evaluation, the gate and monitoring.

The model predicts log(price); every metric takes the true price in LKR and the predicted
log price, and reports errors as percentages, which is how people reason about car prices.
"""

import numpy as np


def score(y_price, pred_log) -> dict:
    y = np.asarray(y_price, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    pred = np.exp(pred_log)
    ape = np.abs(pred - y) / y
    log_y = np.log(y)
    ss_res = float(np.sum((log_y - pred_log) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2)) or 1.0
    return {
        "median_ape": float(np.median(ape)),
        "mape": float(np.mean(ape)),
        "within_20": float(np.mean(ape <= 0.2)),
        "rmse_log": float(np.sqrt(np.mean((log_y - pred_log) ** 2))),
        "r2_log": 1.0 - ss_res / ss_tot,
        "mean_prediction": float(np.mean(pred)),
    }


GATE_KEYS = ["median_ape", "mape", "within_20", "rmse_log", "mean_prediction"]
