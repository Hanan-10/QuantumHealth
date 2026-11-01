
import os
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create required folders
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load Breast Cancer Wisconsin dataset
dataset = load_breast_cancer()

X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

# Scikit-learn uses:
# 0 = Malignant, 1 = Benign
# We convert it to:
# 0 = Benign, 1 = Malignant
y = (dataset.target == 0).astype(int)

# Select four features for quantum model
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area"
]

X = X[selected_features]

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)
print("Selected features:", selected_features)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
pd.DataFrame(
    X_train_scaled,
    columns=selected_features
).to_csv("data/processed/X_train.csv", index=False)

pd.DataFrame(
    X_test_scaled,
    columns=selected_features
).to_csv("data/processed/X_test.csv", index=False)

pd.Series(y_train).to_csv(
    "data/processed/y_train.csv", index=False
)

pd.Series(y_test).to_csv(
    "data/processed/y_test.csv", index=False
)

# Save scaler
joblib.dump(scaler, "models/standard_scaler.pkl")

print("\nPreprocessing completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Benign = 0 | Malignant = 1")
print("All processed files saved successfully!")
