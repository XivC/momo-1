import numpy as np
from collections import deque


def line_search_wolfe(f, grad_f, xk, pk, c1=1e-4, c2=0.9):
    """
    находит оптимальную длину шага alpha
    """
    alpha = 1.0 
    alpha_low = 0
    alpha_high = np.inf
    
    phi_0 = f(xk)
    phi_prime_0 = grad_f(xk).dot(pk)

    max_iter = 20
    for _ in range(max_iter):
        xk_new = xk + alpha * pk
        phi_alpha = f(xk_new)

        if phi_alpha > phi_0 + c1 * alpha * phi_prime_0:
            alpha_high = alpha
            alpha = (alpha_low + alpha_high) / 2
            continue

        grad_new = grad_f(xk_new)
        phi_prime_alpha = grad_new.dot(pk)

        if abs(phi_prime_alpha) <= -c2 * phi_prime_0:
            return alpha # Условия выполнены

        if phi_prime_alpha >= 0:
            alpha_high = alpha_low
            alpha_low = alpha
            alpha = (alpha_low + alpha_high) / 2
            continue
        
        alpha *= 1.5

    return alpha


# --- Реализация алгоритмов оптимизации ---

def bfgs(f, grad_f, x0, max_iter=100, tol=1e-6):
    n = len(x0)
    # Инициализация приближения обратной матрицы Гессе H как единичной матрицы
    H = np.eye(n)
    
    xk = x0.copy()
    history = [xk]

    for k in range(max_iter):
        gk = grad_f(xk)
        if np.linalg.norm(gk) < tol:
            print(f"BFGS сошелся за {k} итераций.")
            return xk, f(xk), history

        pk = -H.dot(gk)

        alpha_k = line_search_wolfe(f, grad_f, xk, pk)
        
        x_next = xk + alpha_k * pk
        
        sk = x_next - xk
        g_next = grad_f(x_next)
        yk = g_next - gk
        
        rho_k = 1.0 / yk.dot(sk)
        I = np.eye(n)
        term1 = I - rho_k * np.outer(sk, yk)
        term2 = I - rho_k * np.outer(yk, sk)
        H = term1.dot(H).dot(term2) + rho_k * np.outer(sk, sk)
        

        xk = x_next
        history.append(xk)
        
    print(f"BFGS не сошелся за {max_iter} итераций.")
    return xk, f(xk), history

def lbfgs(f, grad_f, x0, m=10, max_iter=100, tol=1e-6):
    n = len(x0)
    xk = x0.copy()
    history = [xk]
    
    s_history = deque(maxlen=m)
    y_history = deque(maxlen=m)

    for k in range(max_iter):
        gk = grad_f(xk)
        
        if np.linalg.norm(gk) < tol:
            print(f"L-BFGS (m={m}) сошелся за {k} итераций.")
            return xk, f(xk), history

        q = gk.copy()
        
        alphas = []
        for i in range(len(s_history) - 1, -1, -1):
            rho_i = 1.0 / y_history[i].dot(s_history[i])
            alpha_i = rho_i * s_history[i].dot(q)
            q -= alpha_i * y_history[i]
            alphas.append(alpha_i)
        alphas.reverse()

        r = q 

        for i in range(len(s_history)):
            rho_i = 1.0 / y_history[i].dot(s_history[i])
            beta = rho_i * y_history[i].dot(r)
            r += s_history[i] * (alphas[i] - beta)

        pk = -r 
        alpha_k = line_search_wolfe(f, grad_f, xk, pk)
        
        x_next = xk + alpha_k * pk
        
        sk = x_next - xk
        g_next = grad_f(x_next)
        yk = g_next - gk
        
        s_history.append(sk)
        y_history.append(yk)
        
        xk = x_next
        history.append(xk)

    print(f"L-BFGS (m={m}) не сошелся за {max_iter} итераций.")
    return xk, f(xk), history