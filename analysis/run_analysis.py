import os
import pandas as pd
from snowflake.connector import connect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Use the same credentials you used for dbt (hardcode or better: set environment variables)
conn = connect(
    account=os.environ.get("SNOWFLAKE_ACCOUNT", "SLQAGZU-ORC07712"),
    user=os.environ.get("SNOWFLAKE_USER", "CTAN"),
    password=os.environ.get("SNOWFLAKE_PASSWORD", "L@bingL@bing01"),
    role=os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "WH_DEV"),
    database="CREDIT_DEFAULT",
    schema="MODEL",
)

df = pd.read_sql("SELECT * FROM FEATURE_VIEW_CREDIT_DEFAULT", conn)
conn.close()

y = df["TARGET"].astype(int)
X = df.drop(columns=["TARGET"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

clf = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("AUC:", round(roc_auc_score(y_test, proba), 4))
print("ACC:", round(accuracy_score(y_test, pred), 4))
print("Confusion:", confusion_matrix(y_test, pred).tolist())
