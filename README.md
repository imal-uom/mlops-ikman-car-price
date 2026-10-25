# MLOps: ikman.lk car asking-price model

An end-to-end MLOps system for a car price model trained on ikman.lk advertisements:
versioned data (DVC on DagsHub), tracked experiments and a model registry with a quality gate
(MLflow on DagsHub), a FastAPI service on Azure Container Apps with canary releases, drift
monitoring, and automated retraining.

The model is LightGBM on log(price) with a split-conformal 90% price range. On ads posted after
its training window it has a median error of about 6% and 85% of ads within 20%.

| Folder | What it holds |
|---|---|
| `src/` | cleaning and features, data contract, training, evaluation, quality gate, model registry |
| `app/` | the prediction API: `/predict` returns a price and a range, `/feedback` records real prices |
| `monitoring/` | log export and drift checks |
| `replayer/` | simulated production traffic for the incident drill |
| `infra/` | Azure resources (Bicep) and the bootstrap script |
| `.github/workflows/` | CI, CD, monitoring and retraining |

Common commands: `make test`, `make repro`, `make experiments`, `make serve`.

Data: car advertisements scraped from ikman.lk (21 September to 24 October 2026) with
`src/ikman_scraper.py`; the raw CSV is tracked with DVC. Asking prices, not sale prices.
