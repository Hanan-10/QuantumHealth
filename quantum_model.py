
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

# ------------------------------------
# 1. Load processed dataset
# ------------------------------------

X_train = pd.read_csv("data/processed/X_train.csv").values
X_test = pd.read_csv("data/processed/X_test.csv").values

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Use a small stratified subset for faster quantum training
X_small, _, y_small, _ = train_test_split(
    X_train,
    y_train,
    train_size=100,
    random_state=42,
    stratify=y_train
)

# ------------------------------------
# 2. Encode features into quantum angles
# ------------------------------------

scaler = MinMaxScaler(
    feature_range=(0, np.pi),
    clip=True
)

X_small = scaler.fit_transform(X_small)
X_test = scaler.transform(X_test)

print("Quantum dataset prepared!")
print("Training samples:", len(X_small))
print("Testing samples:", len(X_test))
print("Number of qubits: 4")

# ------------------------------------
# 3. Build quantum feature map
# ------------------------------------

feature_map = zz_feature_map(
    feature_dimension=4,
    reps=1
)

# ------------------------------------
# 4. Build trainable quantum circuit
# ------------------------------------

ansatz = real_amplitudes(
    num_qubits=4,
    reps=1
)

# ------------------------------------
# 5. Initialize quantum simulator
# ------------------------------------

sampler = StatevectorSampler(
    default_shots=512,
    seed=42
)

# ------------------------------------
# 6. Create Variational Quantum Classifier
# ------------------------------------

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=15),
    sampler=sampler
)

print("\nQuantum circuit:")
print(vqc.circuit.draw(output="text"))

# ------------------------------------
# 7. Train quantum classifier
# ------------------------------------

print("\nTraining quantum classifier...")

start = time.time()

vqc.fit(X_small, y_small)

training_time = time.time() - start

print("\nQuantum training completed!")
print("Training time:", round(training_time, 2), "seconds")

# ------------------------------------
# 8. Make quantum predictions
# ------------------------------------

predictions = np.asarray(vqc.predict(X_test)).ravel()

# ------------------------------------
# 9. Evaluate quantum model
# ------------------------------------

print("\n========== QUANTUM VQC RESULTS ==========")

print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("Precision:", round(
    precision_score(y_test, predictions, zero_division=0), 4
))

print("Recall:", round(
    recall_score(y_test, predictions, zero_division=0), 4
))

print("F1 Score:", round(
    f1_score(y_test, predictions, zero_division=0), 4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ------------------------------------
# 10. Fair classical comparison
# ------------------------------------

svm = SVC(kernel="rbf", random_state=42)

svm.fit(X_small, y_small)
joblib.dump(svm, "svm_comparison.pkl")

print("Comparison SVM saved!")

svm_predictions = svm.predict(X_test)
# Save the same-subset SVM for dashboard comparison
import joblib

joblib.dump(svm, "svm_comparison.pkl")

# Save quantum feature scaler
joblib.dump(scaler, "quantum_scaler.pkl")

# Save optimized quantum parameters
np.save("vqc_weights.npy", vqc.weights)

# Save quantum feature scaler
import joblib
joblib.dump(scaler, "quantum_scaler.pkl")

print("Quantum model parameters saved!")

print("\nModels saved successfully!")
print("\n========== SAME-SUBSET SVM COMPARISON ==========")

print("SVM Accuracy:",
      round(accuracy_score(y_test, svm_predictions), 4))

print("VQC Accuracy:",
      round(accuracy_score(y_test, predictions), 4))

print("\nQuantum model training time:",
      round(training_time, 2), "seconds")

print("\nQuantumHealth AI experiment completed!")
