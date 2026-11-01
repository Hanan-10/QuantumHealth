
import numpy as np
import pandas as pd
import joblib
import time

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

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv").values
X_test = pd.read_csv("data/processed/X_test.csv").values

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Select 100 training samples for quantum training
X_small, _, y_small, _ = train_test_split(
    X_train,
    y_train,
    train_size=100,
    stratify=y_train,
    random_state=42
)

# Scale quantum input features between 0 and pi
quantum_scaler = MinMaxScaler(
    feature_range=(0, np.pi),
    clip=True
)

X_quantum = quantum_scaler.fit_transform(X_small)
X_test_quantum = quantum_scaler.transform(X_test)

# Create quantum feature map and ansatz
num_qubits = 4

feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=1
)

ansatz = RealAmplitudes(
    num_qubits=num_qubits,
    reps=1
)

# Quantum simulator
sampler = StatevectorSampler(
    default_shots=512,
    seed=42
)

# Create VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=15),
    sampler=sampler
)

# Train quantum model
print("Training Quantum VQC model...")
start_time = time.time()

vqc.fit(X_quantum, y_small)

training_time = time.time() - start_time

print(f"Quantum training completed in {training_time:.2f} seconds")

# Make predictions
y_pred = vqc.predict(X_test_quantum)
y_pred = np.asarray(y_pred).ravel().astype(int)

# Evaluate quantum model
print("\nQUANTUM VQC RESULTS")
print("-------------------")

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Train comparison SVM on the SAME 100 samples
comparison_svm = SVC(kernel="rbf", random_state=42)
comparison_svm.fit(X_small, y_small)

svm_pred = comparison_svm.predict(X_test)

print("\nSVM COMPARISON RESULTS")
print("----------------------")
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
print(f"VQC Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save model information
np.save("vqc_weights.npy", vqc.weights)

joblib.dump(
    quantum_scaler,
    "quantum_scaler.pkl"
)

joblib.dump(
    comparison_svm,
    "svm_comparison.pkl"
)

print("\nQuantum model files saved successfully!")
