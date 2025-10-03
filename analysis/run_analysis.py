import os
import pandas as pd
from snowflake.connector import connect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np

# Connect using ONLY environment variables (provided by CI; set locally as needed)
conn = connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "WH_DEV"),
    database="CREDIT_DEFAULT",
    schema="MODEL",
)

# Fully qualify the view so we always hit the right object
sql = "SELECT * FROM CREDIT_DEFAULT.MODEL.FEATURE_VIEW_CREDIT_DEFAULT"
df = pd.read_sql(sql, conn)
conn.close()

# Resolve label column case-insensitively (prefer TARGET; fallback to raw name)
cols_upper = {c.upper(): c for c in df.columns}
label = cols_upper.get("TARGET") or cols_upper.get("PAYMENT NEXT MONTH")
if not label:
    raise RuntimeError(f"Label column not found. Columns: {list(df.columns)}")

y = df[label].astype(int)
X = df.drop(columns=[label])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

clf = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

# default 0.50 threshold
pred = (proba >= 0.50).astype(int)
print("AUC:", round(roc_auc_score(y_test, proba), 4))
print("ACC:", round(accuracy_score(y_test, pred), 4))
print("Confusion:", confusion_matrix(y_test, pred).tolist())

# threshold sweep to support policy selection
for t in np.linspace(0.2, 0.8, 7):
    p = (proba >= t).astype(int)
    prec = precision_score(y_test, p, zero_division=0)
    rec = recall_score(y_test, p)
    print(f"th={t:0.2f}  precision={prec:0.3f}  recall={rec:0.3f}")
