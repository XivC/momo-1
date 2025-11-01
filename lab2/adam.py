from typing import Callable, Iterable, List, Tuple, Union

from matrix import Matrix
from vector import euclidean_norm


Number = float | int
PointT = Union[Iterable[Number], Matrix]


def _to_list(point: PointT) -> List[float]:
    if isinstance(point, Matrix):
        rows, cols = point.shape
        if rows == 1:
            return [float(v) for v in point[0]]
        if cols == 1:
            return [float(point[i][0]) for i in range(rows)]

    return [float(v) for v in point]


def finite_difference_gradient(
    func: Callable[[List[float]], float],
    point: PointT,
    step: float = 0.001,
) -> List[float]:
    x = _to_list(point)
    n = len(x)
    grad = [0.0] * n
    fx = None
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += step
        x_minus[i] -= step
        f_plus = float(func(x_plus))
        f_minus = float(func(x_minus))
        grad[i] = (f_plus - f_minus) / (2.0 * step)

    return grad


def adam_optimize(
    func: Callable[[List[float]], float],
    initial_point: PointT,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 0.001,
    max_iters: int = 1000,
    grad_tol: float = 0.001,
    fd_step: float = 0.001,
    callback: Callable[[int, List[float], float, List[float]], None] | None = None,
) -> Tuple[List[float], float, int]:
    "Find min with adam"

    x = _to_list(initial_point)
    n = len(x)
    m = [0.0] * n
    v = [0.0] * n
    t = 0

    for iteration in range(1, max_iters + 1):
        grad = finite_difference_gradient(func, x, step=fd_step)
        grad_norm = euclidean_norm(grad)

        if grad_norm < grad_tol:
            f_val = float(func(x))
            if callback is not None:
                callback(iteration, x, f_val, grad)
            return x, f_val, iteration
        t += 1
        one_minus_beta1_t = 1.0 - beta1**t
        one_minus_beta2_t = 1.0 - beta2**t

        for i in range(n):
            g = grad[i]
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)
            m_hat = m[i] / one_minus_beta1_t
            v_hat = v[i] / one_minus_beta2_t
            x[i] = x[i] - learning_rate * m_hat / ((v_hat ** 0.5) + epsilon)

        f_val = float(func(x))
        if callback is not None:
            callback(iteration, x, f_val, grad)

    f_val = float(func(x))
    return x, f_val, max_iters


if __name__ == "__main__":
    def f(point: List[float]) -> float:
        x, y = point
        return x ** 2 + (y-1) ** 2

    start = Matrix([[0.0, 0.0]])

    p_min, f_min, iters = adam_optimize(
        f,
        start,
        learning_rate=0.05,
        max_iters=2000,
        grad_tol=0.001,
        fd_step=0.001,
    )

    print("Min:", p_min)
    print("f(min):", f_min)
    print("Iters:", iters)
