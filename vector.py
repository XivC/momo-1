from math import copysign
from numbers import Number
from typing import Iterable


def prod_scalar(vector_a: Iterable[Number], vector_b: Iterable[Number]) -> float:

    if len(vector_a) != len(vector_b):
        raise ValueError("Длины векторов должны совпадать для скалярного произведения")
    return float(sum(float(a) * float(b) for a, b in zip(vector_a, vector_b)))


def euclidean_norm(vector: Iterable[Number]) -> float:
    return (prod_scalar(vector, vector))**0.5


def sign(value: float) -> float:

    return 1.0 if value == 0.0 else copysign(1.0, value)

