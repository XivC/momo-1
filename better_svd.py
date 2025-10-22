import numpy as np

def truncated_svd_power_iteration(A, k, num_iterations=15):
    """
    Вычисляет усеченное SVD с использованием степенного метода и дефляции.

    Аргументы:
    A (np.ndarray): Входная матрица формы (m, n).
    k (int): Количество сингулярных компонент для вычисления.
    num_iterations (int): Количество итераций для степенного метода.

    Возвращает:
    U (np.ndarray): Матрица с k левыми сингулярными векторами (m, k).
    S (np.ndarray): Массив с k сингулярными значениями (k,).
    Vt (np.ndarray): Транспонированная матрица с k правыми сингулярными векторами (k, n).
    """
    m, n = A.shape
    A_current = np.copy(A)
    
    # Списки для хранения результатов
    U_list = []
    S_list = []
    Vt_list = []

    for i in range(k):
        # 1. Найти доминантный правый сингулярный вектор v с помощью степенного метода
        # Начинаем со случайного вектора
        v = np.random.randn(n)
        
        # Степенной метод применяется к матрице A.T @ A
        # Это позволяет найти доминантный собственный вектор v
        for _ in range(num_iterations):
            # Умножение на A, затем на A.T, чтобы избежать явного создания A.T @ A
            v = A_current.T @ (A_current @ v)
            v = v / np.linalg.norm(v)

        # 2. Вычислить u и сингулярное значение sigma
        u_unnormalized = A_current @ v
        sigma = np.linalg.norm(u_unnormalized)
        u = u_unnormalized / sigma
        
        # 3. Сохранить найденную сингулярную тройку
        U_list.append(u)
        S_list.append(sigma)
        Vt_list.append(v)
        
        # 4. Выполнить дефляцию: "удалить" найденную компоненту из матрицы
        # Для этого нужно преобразовать u и v в векторы-столбцы и векторы-строки
        A_current = A_current - sigma * np.outer(u, v)

    # 5. Собрать результаты в матрицы
    U = np.array(U_list).T
    S = np.array(S_list)
    Vt = np.array(Vt_list)
    
    return U, S, Vt