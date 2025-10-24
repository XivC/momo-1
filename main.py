from matrix import Matrix
from qr import qr_algorithm
from better_svd import truncated_svd_power_iteration
import numpy as np

if __name__ == "__main__":
    A = Matrix([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
    ])

    vals, _ = qr_algorithm(A)
    print("Eigenvectors:", vals)

    vals2, vecs = qr_algorithm(A, return_eigenvectors=True)
    print("Eigenvalues (sym):", vals)
    print("Eigenvectors (columns):\n", vecs)
    asd = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
    ])
    U, S, Vt = truncated_svd_power_iteration(asd, k=3)
    print("--- Итеративный SVD ---")
    print("U:", U)
    print("Сингулярные значения S:", S)
    print("Vt:", Vt)
