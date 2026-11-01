
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from scipy.optimize import OptimizeResult
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="QuantumHealth AI",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ QuantumHealth AI")
st.subheader("Hybrid Quantum-Classical Disease Classification")

st.markdown("""
An experimental healthcare classification project using
classical machine learning and quantum machine learning.
""")

st.warning(
    "Educational research prototype only. "
    "Predictions are not medical diagnoses."
)

# ---------------------------------
# LOAD TRAINED MODELS
# ---------------------------------

@st.cache_resource
def load_models():

    weights = np.load("vqc_weights.npy")

    quantum_scaler = joblib.load("quantum_scaler.pkl")
    standard_scaler = joblib.load("models/standard_scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")
    svm = joblib.load("svm_comparison.pkl")

    feature_map = zz_feature_map(
        feature_dimension=4,
        reps=1
    )

    ansatz = real_amplitudes(
        num_qubits=4,
        reps=1
    )

    sampler = StatevectorSampler(
        default_shots=512,
        seed=42
    )

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        sampler=sampler
    )

   # Restore trained quantum parameters
    vqc._fit_result = OptimizeResult(x=weights)

    return (
        vqc,
        quantum_scaler,
        standard_scaler,
        imputer,
        svm
    )


try:
    (
        vqc,
        quantum_scaler,
        standard_scaler,
        imputer,
        svm
    ) = load_models()

    st.success("All trained models loaded successfully!")

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("Project Details")

st.sidebar.write("Challenge 02 – Healthcare")
st.sidebar.write("Algorithm: VQC")
st.sidebar.write("Qubits: 4")
st.sidebar.write("Feature Map: ZZFeatureMap")
st.sidebar.write("Ansatz: RealAmplitudes")
st.sidebar.write("Optimizer: COBYLA")
st.sidebar.write("Simulator: StatevectorSampler")

# ---------------------------------
# PATIENT INPUT
# ---------------------------------

st.header("Patient Feature Input")

st.write("Enter the four medical features below.")

col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=30.0
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=1.0,
        max_value=120.0,
        value=35.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=75.0
    )

# ---------------------------------
# PREDICTION
# ---------------------------------

if st.button("Run Quantum Prediction", type="primary"):

    raw_input = pd.DataFrame(
        [[glucose, bmi, age, blood_pressure]],
        columns=[
            "Glucose",
            "BMI",
            "Age",
            "BloodPressure"
        ]
    )

    # Handle zero values as missing measurements
    raw_input = raw_input.replace(0, np.nan)

    # Apply saved preprocessing
    imputed_input = imputer.transform(raw_input)

    # Classical standardized input
    classical_input = standard_scaler.transform(
        imputed_input
    )

    # Quantum angle encoding
    quantum_input = quantum_scaler.transform(
        classical_input
    )

    # Classical prediction
    classical_result = int(
        svm.predict(classical_input)[0]
    )

    # Quantum prediction
    quantum_result = int(
        np.asarray(
            vqc.predict(quantum_input)
        ).ravel()[0]
    )

    st.divider()
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classical SVM")

        if classical_result == 1:
            st.warning("Diabetes-positive class")
        else:
            st.success("Non-diabetic class")

    with col2:
        st.subheader("Quantum VQC")

        if quantum_result == 1:
            st.warning("Diabetes-positive class")
        else:
            st.success("Non-diabetic class")

    if classical_result == quantum_result:
        st.info("Both models produced the same class.")
    else:
        st.info("The two experimental models produced different classes.")

    st.caption(
        "These outputs are experimental model classifications, "
        "not medical diagnoses."
    )

# ---------------------------------
# QUANTUM CIRCUIT
# ---------------------------------

st.divider()
st.header("Quantum Circuit Architecture")

st.write(
    "Our four-qubit Variational Quantum Classifier "
    "uses a ZZFeatureMap and RealAmplitudes ansatz."
)

with st.expander("View quantum circuit"):
    st.code(vqc.circuit.draw(output="text"))

# ---------------------------------
# MODEL PERFORMANCE
# ---------------------------------

st.divider()
st.header("Experimental Model Comparison")

comparison = pd.DataFrame({
    "Model": ["Classical SVM", "Quantum VQC"],
    "Accuracy": [73.38, 44.16]
})

st.bar_chart(
    comparison.set_index("Model")
)

st.caption(
    "Results from the recorded experiment using the same "
    "100-sample training subset and 154-sample test set."
)

st.markdown("---")
st.markdown(
    "**QuantumHealth AI | Quantum Hack Day Kochi | Challenge 02**"
)
