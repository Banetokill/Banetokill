import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json, os

# 1. Load dataset (ensure HEA Phase DataSet v1d.csv is in working directory)
df = pd.read_csv("HEA Phase DataSet v1d.csv", encoding="latin1")

# 2. Define features and target
features = ["Density_calc", "dHmix", "dSmix", "dGmix"]
X = df[features]
y = df['Phase']  # Replace with actual target column name

# 3. Split data
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# 4. Train models
rf = RandomForestClassifier(random_state=random_state)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000, random_state=random_state)
lr.fit(X_train, y_train)

# 5. Evaluate models
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Random Forest Accuracy: {acc_rf:.3f}")
print(f"Logistic Regression Accuracy: {acc_lr:.3f}")
print("\nClassification Report for RF:\n", classification_report(y_test, y_pred_rf))
print("\nClassification Report for LR:\n", classification_report(y_test, y_pred_lr))

# 6. Save models and metrics
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_model.joblib")
joblib.dump(lr, "models/lr_model.joblib")

metrics = {
    'random_forest_accuracy': acc_rf,
    'logistic_regression_accuracy': acc_lr
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)
