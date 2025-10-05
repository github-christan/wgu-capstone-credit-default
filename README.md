![CI](https://github.com/github-christan/wgu-capstone-credit-default/actions/workflows/ci.yml/badge.svg)

# WGU Capstone — Credit Card Default Risk (Snowflake + dbt + CI + ML)

**Goal.** Estimate the **probability of default next month** for credit-card clients and show a lightweight, production-style pipeline:
- **Data engineering:** Snowflake (staging → RAW → dbt models → feature view) + **GitHub Actions** CI.
- **Analytics:** interpretable **Logistic Regression** baseline and **XGBoost** benchmark, with **threshold selection** for policy.

---

## Architecture (quick view)


- **Warehouse:** XS with `AUTO_SUSPEND=60` (keeps credit usage low)  
- **Resource Monitor:** caps spend  
- **CI:** On each push, GitHub Actions runs `dbt build` + `analysis/run_analysis.py`

---

## Results (test set)

| Model                | AUC  | ACC @ 0.50 |
|---------------------|------|------------|
| Logistic Regression | ~0.71| ~0.69      |
| XGBoost             | ~0.78| ~0.76      |

**Policy threshold** (XGBoost): **0.35**  
- Precision ≈ **0.34**, Recall ≈ **0.80**, Accuracy ≈ **0.61**  
- Rationale: for early-warning and outreach, **catching more likely defaulters** (recall) matters; precision can be tuned by raising the cutoff as capacity dictates.

---

## Repository layout
<pre>
.
├─ dbt_project/
│  ├─ models/
│  │  ├─ staging/
│  │  │  └─ stg_credit_default.sql
│  │  ├─ feature_view_credit_default.sql
│  │  └─ models.yml
│  └─ profiles.yml
├─ analysis/
│  ├─ run_analysis.py
│  └─ requirements.txt
├─ .github/
│  ├─ ci/
│  │  └─ profiles.yml
│  └─ workflows/
│     └─ ci.yml
└─ README.md
</pre>


---

## How to run locally

1. **Python env**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   # source .venv/bin/activate

   pip install -r analysis/requirements.txt

2. **Snowflake env vars**
setx SNOWFLAKE_ACCOUNT   "<your_account>"
setx SNOWFLAKE_USER      "<your_user>"
setx SNOWFLAKE_PASSWORD  "<your_password>"
setx SNOWFLAKE_ROLE      "ACCOUNTADMIN"   # or your role
setx SNOWFLAKE_WAREHOUSE "WH_DEV"

3. **dbt (optional local run)**
cd dbt_project
dbt deps
dbt build --select stg_credit_default+ --no-use-colors

4. **Analysis**
cd analysis
python run_analysis.py

## CI / GitHub Actions

Secrets required (Repo → Settings → Secrets & variables → Actions):

SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE

The workflow:

Installs pinned dbt-core==1.10.13 + dbt-snowflake==1.10.2

Runs dbt build (staging + feature view)

Runs analysis/run_analysis.py and prints AUC/ACC + confusion matrix + threshold sweep

## Cost & scale

XS warehouse with auto-suspend; the project ran comfortably under the Snowflake free trial ($397 credits available at start, only $2.95 used).

## Interpretability & governance

Logistic Regression: coefficients → odds-ratios for feature-level directionality.

XGBoost: better ranking power; feature importances and can add SHAP for case explanations.

## Data source

UCI Machine Learning Repository — Default of Credit Card Clients (30,000 rows).
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Used for educational purposes in WGU capstone.
