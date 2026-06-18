import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("data/netflix_churn_sample.csv")

# -------------------------------
# 1. Encode categorical columns
# -------------------------------
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes

# -------------------------------
# 2. Split features & target
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 6. Save model + feature columns
# -------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(X.columns, "features.pkl")

print("\nModel and features saved successfully 🚀")

print(X.columns)