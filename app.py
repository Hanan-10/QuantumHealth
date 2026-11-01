
import streamlit as st
import numpy as np
import joblib
import pandas as pd

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from scipy.optimize import OptimizeResult
from qiskit import ClassicalRegister, transpile
# Page configuration
st.set_page_config(
    page_title="QuantumHealth AI",
    page_icon="⚛️",
    layout="wide"
)


# Premium QuantumHealth AI Theme
st.markdown("""
<style>
.stApp {
    background: #F5F4F8;
    color: #262238;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Typography */
h1 {
    color: #241D35 !important;
    font-weight: 750 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #392D55 !important;
    font-weight: 650 !important;
}

p, label {
    color: #6F687C;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #EAE6F1;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(42, 30, 75, 0.04);
}

[data-testid="stMetricValue"] {
    color: #6745B5;
    font-weight: 750;
}

/* Input fields */
[data-testid="stNumberInput"] input {
    background: #FFFFFF;
    border-radius: 10px;
}

/* Buttons */
 {
    background: #6745B5;
    color: #FFFFFF;
    border: 1px solid #6745B5;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 650;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #50358F;
    border-color: #50358F;
    color: #FFFFFF;.stButton > button
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #E5E0EC;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Charts and images */
[data-testid="stImage"],
[data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"] {
    border-radius: 14px;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# Title
st.title("⚛️ QuantumHealth AI")
st.subheader("Breast Cancer Classification Using Hybrid Quantum-Classical ML")

st.write(
    "An experimental project exploring how classical machine learning "
    "and quantum machine learning can classify breast cancer data."
)

st.warning(
    "Educational prototype only. This application is not a medical "
    "diagnostic tool. Please consult a qualified healthcare professional."
)

# Load saved models
@st.cache_resource
def load_models():
    svm = joblib.load("models/classical_svm.pkl")
    comparison_svm = joblib.load("svm_comparison.pkl")
    scaler = joblib.load("models/standard_scaler.pkl")
    quantum_scaler = joblib.load("quantum_scaler.pkl")
    weights = np.load("vqc_weights.npy")

    # Recreate the quantum circuit
    feature_map = ZZFeatureMap(
        feature_dimension=4,
        reps=1
    )

    ansatz = RealAmplitudes(
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

    # Restore trained weights for Qiskit ML 0.9.1
    vqc._fit_result = OptimizeResult(x=weights)

    return svm, comparison_svm, scaler, quantum_scaler, vqc


try:
    svm, comparison_svm, scaler, quantum_scaler, vqc = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Input section
st.header("Enter Patient Measurements")

col1, col2 = st.columns(2)

with col1:
    radius = st.number_input(
        "Mean Radius",
        min_value=5.0,
        max_value=35.0,
        value=14.0
    )

    texture = st.number_input(
        "Mean Texture",
        min_value=5.0,
        max_value=40.0,
        value=20.0
    )

with col2:
    perimeter = st.number_input(
        "Mean Perimeter",
        min_value=40.0,
        max_value=250.0,
        value=90.0
    )

    area = st.number_input(
        "Mean Area",
        min_value=100.0,
        max_value=2500.0,
        value=600.0
    )

# Prediction
if st.button("Classify", type="primary"):

    input_data = pd.DataFrame(
        [[radius, texture, perimeter, area]],
        columns=[
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area"
        ]
    )

    # Apply the same preprocessing used during training
    scaled_input = scaler.transform(input_data)

    # Classical prediction
    svm_prediction = int(svm.predict(scaled_input)[0])

    # Quantum prediction
    quantum_input = quantum_scaler.transform(scaled_input)
    vqc_raw_prediction = np.asarray(
        vqc.predict(quantum_input)
    )

    vqc_prediction = int(
        np.argmax(vqc_raw_prediction, axis=1)[0]
    )

    # Display results
    st.header("Classification Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.subheader("Classical SVM")
        if svm_prediction == 1:
            st.error("Malignant")
        else:
            st.success("Benign")

    with result_col2:
        st.subheader("Quantum VQC")
        if vqc_prediction == 1:
            st.error("Malignant")
        else:
            st.success("Benign")

    st.info(
        "Benign = 0 | Malignant = 1. "
        "These are experimental model predictions, not medical diagnoses."
    )

    # Feature Insights
    st.divider()
    st.subheader("📊 Patient Feature Insights")

    feature_data = pd.DataFrame({
        "Feature": [
            "Mean Radius",
            "Mean Texture",
            "Mean Perimeter",
            "Mean Area"
        ],
        "Measured Value": [
            radius,
            texture,
            perimeter,
            area
        ]
    })

    st.write(
        "Visual representation of the four measurements "
        "provided for classification."
    )

    st.bar_chart(
        feature_data,
        x="Feature",
        y="Measured Value",
        color="#8064C8"
    )




# Quantum Circuit Visualization
st.divider()
st.header("⚛️ Quantum Circuit")

st.write(
    "The circuit below shows how patient data is encoded, "
    "processed using quantum gates, and measured."
)

# Create a separate display copy of the actual VQC circuit
display_circuit = vqc.feature_map.compose(vqc.ansatz)

# Expand grouped instructions
display_circuit = display_circuit.decompose(reps=5)

# Add classical register for measurements
classical_register = ClassicalRegister(4, "c")
display_circuit.add_register(classical_register)

# Add measurements to all four qubits
display_circuit.measure(
    range(4),
    range(4)
)

# Convert circuit into individual quantum gates
display_circuit = transpile(
    display_circuit,
    basis_gates=["rx", "ry", "rz", "x", "cx"],
    optimization_level=0
)

# Display Qiskit circuit in text diagram format
st.code(
    str(display_circuit.draw(
        output="text",
        fold=80,
        idle_wires=False
    )),
    language="text"
)

st.caption(
    "4 Qubits | ZZFeatureMap | RealAmplitudes | Measurement"
)





# Model Performance Comparison
st.divider()
st.header("📊 Model Performance Comparison")

from sklearn.metrics import accuracy_score

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv").values
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Generate predictions using saved models
svm_predictions = comparison_svm.predict(X_test)

X_test_quantum = quantum_scaler.transform(X_test)
vqc_raw_predictions = np.asarray(
    vqc.predict(X_test_quantum)
)

vqc_predictions = np.argmax(
    vqc_raw_predictions,
    axis=1
)

# Calculate actual breast cancer model accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions) * 100
vqc_accuracy = accuracy_score(y_test, vqc_predictions) * 100

# Display accuracy
col1, col2 = st.columns(2)

with col1:
    st.metric("Classical SVM Accuracy", f"{svm_accuracy:.2f}%")

with col2:
    st.metric("Quantum VQC Accuracy", f"{vqc_accuracy:.2f}%")

# Comparison chart
chart_data = pd.DataFrame({
    "Model": ["Classical SVM", "Quantum VQC"],
    "Accuracy (%)": [svm_accuracy, vqc_accuracy]
})

st.bar_chart(
    chart_data,
    x="Model",
    y="Accuracy (%)"
)

# Footer
st.divider()
st.caption("⚛️ QuantumHealth AI | Quantum Hack Day – Kochi")
