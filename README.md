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
On Windows without `make`: `scripts\dvc_auth.py` (DagsHub login for DVC and MLflow) and
`scripts\run_pipeline.py` (repro, push, gate, promote), both run with `.venv\Scripts\python.exe`.

## Where things live

| Piece | Location |
|---|---|
| Code, CI/CD, monitoring and retraining workflows | this repository |
| Data and model files (DVC), experiments and model registry (MLflow) | https://dagshub.com/imal-uom/mlops-ikman-car-price |
| Container image | `ghcr.io/imal-uom/mlops-ikman-car-price` |
| Live API (Azure Container Apps, resource group `rg-mlops-workshop`, Central India) | https://ca-ikman-car-price.calmgrass-d5ca0c65.centralindia.azurecontainerapps.io |

Try the live service:

```bash
curl https://ca-ikman-car-price.calmgrass-d5ca0c65.centralindia.azurecontainerapps.io/health
curl -X POST https://ca-ikman-car-price.calmgrass-d5ca0c65.centralindia.azurecontainerapps.io/predict \
  -H "Content-Type: application/json" \
  -d '{"brand":"Honda","model":"Vezel","model_year":2018,"mileage_km":60000,"engine_cc":1500,"fuel_type":"Hybrid","transmission":"Automatic","condition":"Used","district":"Colombo"}'
```

Release path: a pull request runs CI (lint, tests, DVC freshness check, quality gate); merging to `main`
runs CD (build and push the image, then, after a reviewer approves the `production` environment, a
0% revision, smoke test, 10% canary, 100%, and promotion of the model to `@champion`). The Monitor
workflow exports the API's prediction logs from Log Analytics every six hours and checks for drift;
Retrain adds labelled production rows, retrains, runs the gate and opens a pull request.

Data: car advertisements scraped from ikman.lk (21 September to 24 October 2026) with
`src/ikman_scraper.py`; the raw CSV is tracked with DVC. Asking prices, not sale prices.
