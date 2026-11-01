
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Download the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, names=columns)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select four features for our quantum model
features = ["Glucose", "BMI", "Age", "BloodPressure"]

X = df[features].copy()
y = df["Outcome"]

# 3. Replace invalid zero measurements with missing values
invalid_columns = ["Glucose", "BMI", "BloodPressure"]

X[invalid_columns] = X[invalid_columns].replace(0, np.nan)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Handle missing values using training data only
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# Save preprocessing objects
os.makedirs("models", exist_ok=True)

joblib.dump(imputer, "models/imputer.pkl")
# 6. Normalize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "models/standard_scaler.pkl")

print("Preprocessing objects saved!")


# 7. Save processed data
os.makedirs("data/processed", exist_ok=True)

pd.DataFrame(X_train, columns=features).to_csv(
    "data/processed/X_train.csv", index=False
)
pd.DataFrame(X_test, columns=features).to_csv(
    "data/processed/X_test.csv", index=False
)

pd.Series(y_train).to_csv(
    "data/processed/y_train.csv", index=False
)
pd.Series(y_test).to_csv(
    "data/processed/y_test.csv", index=False
)

print("\nPreprocessing completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Quantum features:", features)
print("\nAll processed files saved successfully!")
