import os
import numpy as np
import pandas as pd
from snowflake.connector import connect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score
)
from xgboost import XGBClassifier

# --- Connect with env vars ---
conn = connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "WH_DEV"),
    database="CREDIT_DEFAULT",
    schema="MODEL",
)

# --- Load features/view (fully qualified) ---
sql = "SELECT * FROM CREDIT_DEFAULT.MODEL.FEATURE_VIEW_CREDIT_DEFAULT"
df = pd.read_sql(sql, conn)
conn.close()

# --- Resolve label column robustly ---
cols_upper = {c.upper(): c for c in df.columns}
label = cols_upper.get("TARGET") or cols_upper.get("PAYMENT NEXT MONTH")
if not label:
    raise RuntimeError(f"Label column not found. Columns: {list(df.columns)}")

# Optional: ensure numeric & drop any accidental nulls
df = df.dropna()
y = df[label].astype(int)
X = df.drop(columns=[label])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ========= Logistic Regression =========
lr = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_pred  = (lr_proba >= 0.50).astype(int)

print("=== Logistic Regression ===")
print("AUC:", round(roc_auc_score(y_test, lr_proba), 4))
print("ACC:", round(accuracy_score(y_test, lr_pred), 4))
print("Confusion:", confusion_matrix(y_test, lr_pred).tolist())
for t in np.linspace(0.2, 0.8, 7):
    p = (lr_proba >= t).astype(int)
    print(f"th={t:0.2f}  precision={precision_score(y_test, p, zero_division=0):0.3f}  "
          f"recall={recall_score(y_test, p):0.3f}")

# ========= XGBoost benchmark =========
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / max(pos, 1)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    n_jobs=2,
    random_state=42,
    tree_method="hist",
    # early_stopping_rounds=30,  # uncomment if you want faster CI
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_pred  = (xgb_proba >= 0.50).astype(int)

print("\n=== XGBoost (benchmark) ===")
print("AUC:", round(roc_auc_score(y_test, xgb_proba), 4))
print("ACC:", round(accuracy_score(y_test, xgb_pred), 4))
print("Confusion:", confusion_matrix(y_test, xgb_pred).tolist())
for t in np.linspace(0.2, 0.8, 7):
    p = (xgb_proba >= t).astype(int)
    print(f"th={t:0.2f}  precision={precision_score(y_test, p, zero_division=0):0.3f}  "
          f"recall={recall_score(y_test, p):0.3f}")

# --- Compact comparison block (nice for CI summary) ---
lr_auc  = roc_auc_score(y_test, lr_proba)
xgb_auc = roc_auc_score(y_test, xgb_proba)
lr_acc  = accuracy_score(y_test, lr_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
print("\n=== Summary ===")
print(f"LR  : AUC={lr_auc:0.4f}  ACC={lr_acc:0.4f}")
print(f"XGB : AUC={xgb_auc:0.4f}  ACC={xgb_acc:0.4f}")
