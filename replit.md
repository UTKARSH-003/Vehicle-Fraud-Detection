# ClaimSignal

ClaimSignal compares vehicle-insurance fraud classifiers and provides investigator-facing claim triage scores from the supplied historical dataset.

## Run & Operate

- `pnpm --filter @workspace/claim-fraud run dev` — run the web interface
- `pnpm --filter @workspace/api-server run dev` — run the API and its Flask model service
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- Local install and single-process Flask serving instructions: `artifacts/claim-fraud/README.md`
- Python dependencies: `artifacts/claim-fraud/requirements.txt`

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9, React + Vite
- API gateway: Express 5; model API: Flask
- Machine learning: pandas, scikit-learn, XGBoost, joblib
- API codegen: Orval (from OpenAPI spec)
- Build: Vite and esbuild

## Where things live

- `artifacts/claim-fraud/fraud_ml/` — Python preprocessing, training, model cache, inference, and Flask API
- `artifacts/claim-fraud/data/fraud_oracle.csv` — supplied training data
- `artifacts/claim-fraud/src/` — React web interface
- `lib/api-spec/openapi.yaml` — API contract and generated React Query hooks

## Architecture decisions

- Compare Logistic Regression, Random Forest, and XGBoost with stratified training, validation, and held-out test splits.
- Use class-weighted training; select thresholds and the winning model using validation F0.5 to emphasize precision and reduce investigator false alerts.
- Exclude policy and representative identifiers plus demographic fields from the model inputs.
- Scores are triage signals only. The historical dataset has no claim amount or policy-tenure field.

## Product

The overview reports dataset composition and held-out precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices. Investigators can enter dataset-defined claim fields and receive a probability score, threshold-based triage recommendation, and limitations.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The first model-service startup trains and caches estimators if the data-matched model bundle is missing.
- The generated model cache is ignored by Git and rebuilt when the dataset fingerprint or model version changes.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
