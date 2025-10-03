import os
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

engine = create_engine(URL(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "WH_DEV"),
    database="CREDIT_DEFAULT",
    schema="MODEL",
))

df = pd.read_sql("SELECT * FROM FEATURE_VIEW_CREDIT_DEFAULT", engine)
engine.dispose()

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
