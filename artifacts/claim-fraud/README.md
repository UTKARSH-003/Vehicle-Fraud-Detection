# ClaimSignal — Vehicle Insurance Claim Fraud Triage

ClaimSignal compares Logistic Regression, Random Forest, and XGBoost on the
included vehicle-insurance claims dataset, then provides an investigator-facing
risk score and triage recommendation. Scores support review; they are not
findings of fraud or claim decisions.

## Run locally

Requirements: Python 3.10+ and Node.js with pnpm.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r artifacts/claim-fraud/requirements.txt
pnpm install
PORT=5000 BASE_PATH=/ pnpm --filter @workspace/claim-fraud run build
PYTHONPATH=artifacts/claim-fraud FRAUD_SERVICE_PORT=5000 python -m fraud_ml.service
```

Open `http://127.0.0.1:5000`. The first launch trains the three classifiers and
caches the fitted preprocessing/model bundle under `artifacts/claim-fraud/models/`.
Subsequent launches reuse the bundle unless the CSV or model version changes.
To train explicitly after activating the virtual environment:

```bash
PYTHONPATH=artifacts/claim-fraud python -m fraud_ml.training
```

For frontend iteration instead of the built-in Flask-served page, run the Vite
workflow from Replit and keep the API server running. The API server starts the
local Flask model service and proxies `/api/fraud/*` requests to it.

## Data and evaluation

- Target: `FraudFound_P` (`1` = fraud, `0` = non-fraud).
- The CSV is split stratified into training and held-out test data. Training is
  split again for threshold selection; the test split is not used to choose a
  threshold or model.
- Missing numeric values are median-imputed and scaled. Missing categorical
  values are most-frequent-imputed and one-hot encoded. Unknown categories are
  ignored by the fitted encoder.
- Logistic Regression and Random Forest use balanced class weights. XGBoost
  uses the negative-to-positive class ratio as `scale_pos_weight`.
- Each candidate's decision threshold is selected on validation F1, with
  precision as the tie-breaker. The selected candidate is the one with the
  best validation F1, then precision, then recall. Precision, recall, F1,
  ROC-AUC, PR-AUC, and confusion matrices are measured on the held-out test set.
- `PolicyNumber`, `RepNumber`, sex, age, marital status, and policyholder age
  are excluded from model inputs. The model form is generated from the remaining
  source columns.
- The dataset has no claim amount or policy-tenure column. Those fields are not
  represented by proxies and are not used in scoring.

## Project structure

- `fraud_ml/preprocessing.py` — CSV cleaning, missing-value strategy, scaling,
  encoding, and dynamic form-field metadata.
- `fraud_ml/training.py` — stratified splits, class-imbalance handling, model
  comparison, threshold selection, and test metrics.
- `fraud_ml/model_manager.py` — data-versioned model cache and safe atomic writes.
- `fraud_ml/inference.py` — dataset/model summaries and claim scoring.
- `fraud_ml/web.py` — Flask JSON API and local static frontend serving.
- `fraud_ml/service.py` — model warm-up and local Flask server entrypoint.
- `data/fraud_oracle.csv` — supplied training dataset.

## Important limitations

This is a historical benchmark, not a validated production insurance model.
Its score must not be used by itself to deny, delay, or reduce a claim. Evaluate
performance and subgroup error rates on current, representative, legally
approved data before operational use, and retain human review and an appeal path.
