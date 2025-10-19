from typing import Tuple, Union

from matrix import Matrix
from vector import euclidean_norm, sign


def qr_decomposition_householder(matrix: "Matrix") -> Tuple["Matrix", "Matrix"]:
    """
    QR-разложение матрицы методом отражений Хаусхолдера

    :return: tuple(Q, R), где Q — (n x n), R — (n x m).
    """

    r_matrix: Matrix = matrix.copy()
    row_count, col_count = r_matrix.shape
    q_matrix: Matrix = Matrix.eye(row_count)

    for pivot_index in range(min(row_count, col_count) - 1):
        # Берём столбец pivot_index, начиная с строки pivot_index
        column_tail: list[float] = [r_matrix[row_index][pivot_index] for row_index in range(pivot_index, row_count)]
        column_tail_norm: float = euclidean_norm(column_tail)
        if column_tail_norm == 0.0:
            continue

        unit_first: list[float] = [0.0] * (row_count - pivot_index)
        unit_first[0] = 1.0

        alpha_value: float = sign(column_tail[0]) * column_tail_norm

        # u = x + alpha * e1
        householder_raw: list[float] = [
            column_tail[idx] + alpha_value * unit_first[idx] for idx in range(row_count - pivot_index)
        ]
        householder_norm: float = euclidean_norm(householder_raw)
        if householder_norm == 0.0:
            continue

        # v = u / ||u||
        householder_vector: list[float] = [component / householder_norm for component in householder_raw]

        reflector_full: list[float] = [0.0] * row_count
        for local_index, value in enumerate(householder_vector):
            reflector_full[pivot_index + local_index] = value

        # Матрица отражения H = I - 2 * v v^T
        identity_matrix: Matrix = Matrix.eye(row_count)
        outer_product: Matrix = Matrix.outer(reflector_full, reflector_full)
        householder_matrix: Matrix = identity_matrix - (2.0 * outer_product)

        r_matrix = householder_matrix * r_matrix
        q_matrix = q_matrix * householder_matrix

    return q_matrix, r_matrix


def qr_algorithm(
    matrix: "Matrix",
    max_iterations: int = 1000,
    tolerance: float = 1e-12,
    return_eigenvectors: bool = False
) -> Tuple[list[float], Matrix | None]:
    """
    QR-алгоритм без сдвигов для приближённого нахождения собственных значений
    """
    if not matrix.is_square():
        raise ValueError("QR-алгоритм требует квадратную матрицу")

    size, _ = matrix.shape
    iterated_matrix: Matrix = matrix.copy()
    eigenvectors_matrix: Matrix = Matrix.eye(size)

    for _iteration_index in range(max_iterations):
        q_matrix, r_matrix = qr_decomposition_householder(iterated_matrix)
        iterated_matrix = r_matrix * q_matrix

        eigenvectors_matrix = eigenvectors_matrix * q_matrix

        sum_of_squares_offdiag: float = 0.0
        for row_index in range(size):
            for col_index in range(0, row_index):
                value = iterated_matrix[row_index][col_index]
                sum_of_squares_offdiag += value * value
        offdiag_norm: float = sum_of_squares_offdiag**0.5
        if offdiag_norm < tolerance:
            break

    eigenvalues: list[float] = iterated_matrix.diagonal()

    if return_eigenvectors and matrix.is_symmetric():
        normalized_vectors: Matrix = eigenvectors_matrix.copy()
        for col_index in range(size):
            column_values: list[float] = [normalized_vectors[row_index][col_index] for row_index in range(size)]
            column_norm: float = euclidean_norm(column_values)
            if column_norm != 0.0:
                inv_norm: float = 1.0 / column_norm
                for row_index in range(size):
                    normalized_vectors[row_index][col_index] *= inv_norm
        return eigenvalues, normalized_vectors

    return eigenvalues, None
