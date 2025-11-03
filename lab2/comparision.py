from typing import Callable
import time
import numpy as np
import matplotlib.pyplot as plt

from lab2.adam import adam_optimize
from lab2.utils import bfgs, lbfgs

TEST_FUNCTIONS: dict[str, tuple[Callable[[np.ndarray], float], np.ndarray]] = {
    "ShiftedQuadratic": (
        lambda x: float(np.dot(x - 3.0, x - 2.0)),
        np.full(5, 2.0)
    ),
    "Griewank": (
        lambda x: float(
            1.0 + np.sum(x ** 2) / 4000.0
            - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        ),
        np.zeros(5)
    ),
}

def numeric_gradient(func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x1 = x.copy(); x1[i] += h
        x2 = x.copy(); x2[i] -= h
        g[i] = (func(x1) - func(x2)) / (2.0 * h)
    return g


def run_bfgs_timed(func: Callable[[np.ndarray], float], x0: np.ndarray, tol: float) -> tuple[float, int]:
    start = time.perf_counter()
    x_opt, f_opt, history = bfgs(
        f=lambda x: func(x),
        grad_f=lambda x: numeric_gradient(func, x),
        x0=x0,
        tol=tol,
        max_iter=3000,
    )
    end = time.perf_counter()
    iters = len(history)
    return end - start, iters


def run_lbfgs_timed(func: Callable[[np.ndarray], float], x0: np.ndarray, tol: float, m: int = 5) -> tuple[float, int]:
    start = time.perf_counter()
    x_opt, f_opt, history = lbfgs(
        f=lambda x: func(x),
        grad_f=lambda x: numeric_gradient(func, x),
        x0=x0,
        m=m,
        max_iter=3000,
        tol=tol,
    )
    end = time.perf_counter()
    iters = len(history)
    return end - start, iters


def run_adam_timed(func: Callable[[np.ndarray], float], x0: np.ndarray, tol: float) -> tuple[float, int]:
    start = time.perf_counter()
    x_opt, f_opt, iters = adam_optimize(
        func=lambda lst: func(np.array(lst, dtype=float)),
        initial_point=list(x0),
        grad_tol=tol,
        max_iters=3000,
        fd_step=1e-4,
    )
    end = time.perf_counter()
    return end - start, iters


def main() -> None:

    tol_powers = [n for n in range(1, 10)]
    tolerances = [10**(-n) for n in tol_powers]

    for fname, (f, x_min) in TEST_FUNCTIONS.items():
        x0 = np.array([3.0, -1.5, 0.7, 2.1, -0.3])

        bfgs_times: list[float] = []
        lbfgs_times: list[float] = []
        adam_times: list[float] = []

        for tol in tolerances:
            t_bfgs, _ = run_bfgs_timed(f, x0, tol)
            t_lbfgs, _ = run_lbfgs_timed(f, x0, tol, m=5)
            t_adam, _ = run_adam_timed(f, x0, tol)

            bfgs_times.append(t_bfgs)
            lbfgs_times.append(t_lbfgs)
            adam_times.append(t_adam)

        plt.figure(figsize=(6, 4))
        plt.plot(tol_powers, bfgs_times, marker="o", label="BFGS")
        plt.plot(tol_powers, lbfgs_times, marker="o", label="L-BFGS(m=5)")
        plt.plot(tol_powers, adam_times, marker="o", label="Adam")
        plt.yscale("log")
        plt.xlabel("требуемая точность (10^n)")
        plt.ylabel("время, с")
        plt.title(f"функция: {fname}")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
