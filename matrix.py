from __future__ import annotations
from typing import List, Tuple, Iterable, Union, Any
from copy import deepcopy
from math import sqrt


Number = Union[int, float]


class Matrix:
    def __init__(self, values: Iterable[Iterable[Number]]) -> None:
        if (
            not values
            or not isinstance(values, (list, tuple))
            or not isinstance(values[0], (list, tuple))
        ):
            raise ValueError("Данные матрицы должны быть непустым списком списков")

        row_count: int = len(values)
        col_count: int = len(values[0])

        for row in values:
            if len(row) != col_count:
                raise ValueError("Все строки матрицы должны иметь одинаковую длину")

        self._values: List[List[float]] = [
            [float(element) for element in row] for row in values
        ]
        self._row_count: int = row_count
        self._col_count: int = col_count

    @staticmethod
    def zeros(row_count: int, col_count: int) -> "Matrix":
        return Matrix([[0.0] * col_count for _ in range(row_count)])

    @staticmethod
    def eye(size: int) -> "Matrix":
        identity = Matrix.zeros(size, size)
        for index in range(size):
            identity._values[index][index] = 1.0
        return identity

    @property
    def shape(self) -> Tuple[int, int]:
        return self._row_count, self._col_count

    def copy(self) -> "Matrix":
        return Matrix(deepcopy(self._values))

    def is_square(self) -> bool:
        return self._row_count == self._col_count

    def is_symmetric(self, tolerance: float = 1e-12) -> bool:
        if not self.is_square():
            return False
        size: int = self._row_count
        for row_index in range(size):
            for col_index in range(row_index + 1, size):
                if abs(self._values[row_index][col_index] - self._values[col_index][row_index]) > tolerance:
                    return False
        return True

    def __getitem__(self, row_index: int) -> List[float]:
        return self._values[row_index]

    def __setitem__(self, row_index: int, new_row: Iterable[Number]) -> None:

        if len(new_row) != self._col_count:
            raise ValueError("Длина строки не совпадает с числом столбцов")

        self._values[row_index] = [float(element) for element in new_row]

    def __add__(self, other: Union["Matrix", Number]) -> "Matrix":
        if isinstance(other, Matrix):
            self_rows, self_cols = self.shape
            other_rows, other_cols = other.shape
            if (self_rows, self_cols) != (other_rows, other_cols):
                raise ValueError("Размеры матриц не совпадают для сложения")
            return Matrix(
                [
                    [self._values[row_index][col_index] + other._values[row_index][col_index] for col_index in range(self_cols)]
                    for row_index in range(self_rows)
                ]
            )
        elif isinstance(other, (int, float)):
            self_rows, self_cols = self.shape
            return Matrix(
                [
                    [self._values[row_index][col_index] + float(other) for col_index in range(self_cols)]
                    for row_index in range(self_rows)
                ]
            )
        else:
            raise TypeError("Поддерживается сложение только с Matrix или числом")

    def __sub__(self, other: Union["Matrix", Number]) -> "Matrix":
        if isinstance(other, Matrix):
            self_rows, self_cols = self.shape
            other_rows, other_cols = other.shape
            if (self_rows, self_cols) != (other_rows, other_cols):
                raise ValueError("Размеры матриц не совпадают для вычитания")
            return Matrix(
                [
                    [self._values[row_index][col_index] - other._values[row_index][col_index] for col_index in range(self_cols)]
                    for row_index in range(self_rows)
                ]
            )
        elif isinstance(other, (int, float)):
            self_rows, self_cols = self.shape
            return Matrix(
                [
                    [self._values[row_index][col_index] - float(other) for col_index in range(self_cols)]
                    for row_index in range(self_rows)
                ]
            )
        else:
            raise TypeError("Поддерживается вычитание только с Matrix или числом")

    def __rmul__(self, other: Number) -> "Matrix":
        return self.__mul__(other)

    def __mul__(self, other: Union["Matrix", Number]) -> "Matrix":
        if isinstance(other, (int, float)):
            row_count, col_count = self.shape
            factor = float(other)
            return Matrix(
                [
                    [self._values[row_index][col_index] * factor for col_index in range(col_count)]
                    for row_index in range(row_count)
                ]
            )
        elif isinstance(other, Matrix):
            left_rows, left_cols = self.shape
            right_rows, right_cols = other.shape
            if left_cols != right_rows:
                raise ValueError("Формы матриц несовместимы для умножения")
            result = Matrix.zeros(left_rows, right_cols)
            for row_index in range(left_rows):
                for common_index in range(left_cols):
                    left_value = self._values[row_index][common_index]
                    if left_value == 0.0:
                        continue
                    result_row = result._values[row_index]
                    other_row = other._values[common_index]
                    for col_index in range(right_cols):
                        result_row[col_index] += left_value * other_row[col_index]
            return result
        else:
            raise TypeError("Поддерживается умножение только с Matrix или числом")

    def transposed(self) -> "Matrix":
        row_count, col_count = self.shape
        return Matrix([[self._values[row_index][col_index] for row_index in range(row_count)] for col_index in range(col_count)])

    @staticmethod
    def outer(vector_left: Iterable[Number], vector_right: Iterable[Number]) -> "Matrix":
        return Matrix(
            [
                [float(vector_left[row_index]) * float(vector_right[col_index]) for col_index in range(len(vector_right))]
                for row_index in range(len(vector_left))
            ]
        )

    def norm(self) -> float:
        squared_sum: float = 0.0
        for row in self._values:
            for value in row:
                squared_sum += value * value
        return sqrt(squared_sum)

    def diagonal(self) -> List[float]:
        if not self.is_square():
            raise ValueError("Диагональ определена только для квадратных матриц")
        return [self._values[index][index] for index in range(self._row_count)]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        formatted_rows: List[str] = []
        for row in self._values:
            formatted_rows.append("[" + "  ".join(f"{value: .6g}" for value in row) + "]")
        return "\n".join(formatted_rows)