
import qiskit
import qiskit_aer
import qiskit_machine_learning
import sklearn

print("Qiskit:", qiskit.__version__)
print("Qiskit Aer:", qiskit_aer.__version__)
print("Qiskit Machine Learning:", qiskit_machine_learning.__version__)
print("Scikit-learn:", sklearn.__version__)

from qiskit_machine_learning.algorithms import VQC

print("\nVQC import successful!")
