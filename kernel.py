from numbers import Number
from typing import Callable, Iterable, List, Tuple, Union
from math import exp, tanh

from matrix import Matrix
from qr import qr_algorithm, eigen_decomposition_qr
from vector import validate_vectors_same_length, prod_scalar, euclidean_norm


def polynomial_kernel(
    vector_a: Iterable[Number],
    vector_b: Iterable[Number],
    degree: int = 2,
    gamma: Union[None, float] = None,
    coef0: float = 1.0,
) -> float:
    dim = validate_vectors_same_length(vector_a, vector_b)
    if degree < 1:
        raise ValueError("Степень полиномиального ядра должна быть положительным целым числом")
    if gamma is None:
        gamma = 1.0 / float(dim)
    return (gamma * prod_scalar(vector_a, vector_b) + float(coef0)) ** int(degree)


def rbf_kernel(
    vector_a: Iterable[Number],
    vector_b: Iterable[Number],
    gamma: Union[None, float] = None,
) -> float:
    dim = validate_vectors_same_length(vector_a, vector_b)
    if gamma is None:
        gamma = 1.0 / float(dim)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    norm_a2 = euclidean_norm(vector_a) ** 2
    norm_b2 = euclidean_norm(vector_b) ** 2
    dot_ab = prod_scalar(vector_a, vector_b)
    squared_distance = norm_a2 + norm_b2 - 2.0 * dot_ab
    return float(exp(-float(gamma) * squared_distance))


def sigmoid_kernel(
    vector_a: Iterable[Number],
    vector_b: Iterable[Number],
    gamma: Union[None, float] = None,
    coef0: float = 0.0,
) -> float:
    dim = validate_vectors_same_length(vector_a, vector_b)
    if gamma is None:
        gamma = 1.0 / float(dim)
    return float(tanh(float(gamma) * prod_scalar(vector_a, vector_b) + float(coef0)))


def compute_kernel_matrix(
    matrix: Matrix,
    kernel: Callable[..., float],
    **kernel_params: Number,
) -> Matrix:
    matrix = matrix.copy()
    matrix_length = matrix.shape[0]
    if matrix_length == 0:
        raise ValueError("Нельзя строить матрицу ядра по пустому набору данных")
    gram = Matrix.zeros(matrix_length, matrix_length)

    for i in range(matrix_length):
        gram[i][i] = float(kernel(matrix[i], matrix[i], **kernel_params))
        for j in range(i + 1, matrix_length):
            kij: float = float(kernel(matrix[i], matrix[j], **kernel_params))
            gram[i][j] = kij
            gram[j][i] = kij
    return gram


def kernel_pca(
    matrix: Matrix,
    kernel_func: Callable[..., float],
    n_components: int = 2,
    **kernel_params
) -> Tuple[Matrix, List[float]]:
    """
    Kernel PCA impl
    :Return:
      - матрицу проекций (n_samples x n_components)
      - список собственных значений
    """
    matrix = matrix.copy()
    K = compute_kernel_matrix(matrix, kernel_func, **kernel_params)
    matrix_length, _ = K.shape

    ones = Matrix([[1.0 / matrix_length] * matrix_length for _ in range(matrix_length)])
    K_centered = K - ones * K - K * ones + ones * K * ones

    eigenvalues, eigenvectors = qr_algorithm(
        K_centered.copy(),
        max_iterations=1000,
        tolerance=1e-12,
        return_eigenvectors=True
    )

    eig_pairs = sorted(zip(eigenvalues, list(zip(*eigenvectors._values))), key=lambda x: x[0], reverse=True)
    sorted_values, sorted_vectors = zip(*eig_pairs)

    selected_vectors = [list(vec) for vec in sorted_vectors[:n_components]]
    selected_values = list(sorted_values[:n_components])

    Z = Matrix.zeros(matrix_length, n_components)
    for j in range(n_components):
        sqrt_lambda = (selected_values[j])**0.5 if selected_values[j] > 1e-15 else 1.0
        for i in range(matrix_length):
            Z[i][j] = sum(K_centered[i][k] * selected_vectors[j][k] for k in range(matrix_length)) / sqrt_lambda

    return Z, selected_values[:n_components]


if __name__ == "__main__":
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    X = Matrix(X)
    y: List[int] = [0, 1, 1, 0]

    K_poly: Matrix = compute_kernel_matrix(
        X,
        polynomial_kernel,
        degree=3,
        coef0=1.0
    )
    vals_poly, vecs_poly = eigen_decomposition_qr(K_poly)

    K_rbf: Matrix = compute_kernel_matrix(
        X,
        rbf_kernel,
        gamma=1.0
    )
    vals_rbf, vecs_rbf = eigen_decomposition_qr(K_rbf)

    K_sig: Matrix = compute_kernel_matrix(
        X,
        sigmoid_kernel,
        gamma=1.0,
        coef0=0.0
    )
    vals_sig, vecs_sig = eigen_decomposition_qr(K_sig)

    print("Полиномиальное ядро")
    print("Матрица K:\n", K_poly, sep="")
    print("Собственные значения:", [f"{v:.3g}" for v in vals_poly])
    print("Собственные векторы:\n", vecs_poly, sep="")
    print()

    print("RBF ядро")
    print("Матрица K:\n", K_rbf, sep="")
    print("Собственные значения:", [f"{v:.3g}" for v in vals_rbf])
    print("Собственные векторы:\n", vecs_rbf, sep="")
    print()

    print("Сигмоидальное ядро")
    print("Матрица K:\n", K_sig, sep="")
    print("Собственные значения:", [f"{v:.3g}" for v in vals_sig])
    print("Собственные векторы:\n", vecs_sig, sep="")
    print()

    print("PCA")
    print()
    for kernel in (rbf_kernel, sigmoid_kernel, polynomial_kernel):
        print(f"Ядро: {kernel.__name__}")
        Z, lambdas = kernel_pca(X, rbf_kernel, n_components=2, gamma=1.0)

        print("Собственные значения:", [f"{v:.6g}" for v in lambdas])
        print("Проекции точек (в ядровом пространстве):")
        print(Z)
