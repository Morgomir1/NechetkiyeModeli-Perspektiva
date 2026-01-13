# -*- coding: utf-8 -*-
"""
Лабораторная работа 10: Марковские цепи для студентов
Определение матрицы переходных вероятностей и решение векторно-матричного уравнения (10)-(11)
"""

import numpy as np

def build_transition_matrix(transitions):
    """
    Строит матрицу переходных вероятностей на основе данных о переходах
    
    transitions: словарь вида {(i, j): count} - количество переходов из состояния i в состояние j
    """
    # Определяем количество состояний
    states = set()
    for (i, j) in transitions.keys():
        states.add(i)
        states.add(j)
    n = len(states)
    states = sorted(list(states))
    
    # Создаем матрицу подсчета переходов
    count_matrix = np.zeros((n, n))
    for (i, j), count in transitions.items():
        i_idx = states.index(i)
        j_idx = states.index(j)
        count_matrix[i_idx, j_idx] = count
    
    # Нормализуем по строкам для получения вероятностей
    P = np.zeros((n, n))
    for i in range(n):
        row_sum = count_matrix[i, :].sum()
        if row_sum > 0:
            P[i, :] = count_matrix[i, :] / row_sum
        else:
            P[i, i] = 1.0  # Если нет переходов, остаемся в том же состоянии
    
    return P, states

def solve_stationary_distribution(M):
    """
    Решает векторно-матричное уравнение (10)-(11) для нахождения стационарного распределения
    
    Уравнение (10): p̅ × (M - E) = 0
    где p̅ - вектор-строка установившихся состояний (p̅₁; p̅₂; p̅₃; p̅₄)
    M - матрица переходов
    E - единичная матрица
    
    Уравнение (11): Σⱼ p̅ⱼ = 1 (нормировочное условие)
    
    Возвращает вектор-строку стационарных вероятностей p̅
    """
    n = M.shape[0]
    E = np.eye(n)  # Единичная матрица
    
    # Уравнение (10): p̅ × (M - E) = 0
    # Преобразуем в систему: (M - E)ᵀ p̅ᵀ = 0
    # Или: (Mᵀ - E) p̅ᵀ = 0
    A = M.T - E
    
    # Система линейно-зависима, поэтому заменяем последнее уравнение
    # на нормировочное условие (11): Σⱼ p̅ⱼ = 1
    A[-1, :] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1.0
    
    # Решаем систему линейных уравнений
    try:
        p_bar = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Если система вырождена, используем метод с собственными векторами
        # Ищем собственный вектор Mᵀ, соответствующий собственному значению 1
        eigenvalues, eigenvectors = np.linalg.eig(M.T)
        # Находим индекс собственного значения, наиболее близкого к 1
        idx = np.argmin(np.abs(np.real(eigenvalues) - 1.0))
        p_bar = np.real(eigenvectors[:, idx])
        # Нормализуем согласно уравнению (11)
        p_bar = p_bar / p_bar.sum()
    
    # Убеждаемся, что все вероятности неотрицательны
    p_bar = np.maximum(p_bar, 0)
    # Нормализуем согласно уравнению (11)
    p_bar = p_bar / p_bar.sum()
    
    return p_bar

# Данные для студента 1
# Предположим, что состояния - это оценки: 1 (отлично), 2 (хорошо), 3 (удовлетворительно), 4 (неудовлетворительно)
# Данные о переходах между оценками
student1_transitions = {
    (1, 1): 15, (1, 2): 5, (1, 3): 2, (1, 4): 0,
    (2, 1): 3, (2, 2): 12, (2, 3): 4, (2, 4): 1,
    (3, 1): 1, (3, 2): 3, (3, 3): 8, (3, 4): 2,
    (4, 1): 0, (4, 2): 1, (4, 3): 2, (4, 4): 5
}

# Данные для студента 2
student2_transitions = {
    (1, 1): 18, (1, 2): 4, (1, 3): 1, (1, 4): 0,
    (2, 1): 4, (2, 2): 14, (2, 3): 3, (2, 4): 0,
    (3, 1): 2, (3, 2): 4, (3, 3): 9, (3, 4): 1,
    (4, 1): 0, (4, 2): 2, (4, 3): 3, (4, 4): 4
}

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА 10")
print("Определение матрицы переходных вероятностей и решение векторно-матричного уравнения")
print("=" * 70)

# Обработка данных для студента 1
print("\n" + "=" * 70)
print("СТУДЕНТ 1")
print("=" * 70)

P1, states1 = build_transition_matrix(student1_transitions)
print(f"\nСостояния: {states1}")
print("\nМатрица переходных вероятностей P₁:")
print(P1)

# Решение уравнения (10)-(11) для студента 1
p_bar1 = solve_stationary_distribution(P1)
print("\nСтационарное распределение (решение уравнений (10)-(11)):")
for i, state in enumerate(states1):
    print(f"p̅₁({state}) = {p_bar1[i]:.6f}")

print(f"\nПроверка уравнения (11): Σⱼ p̅ⱼ = {p_bar1.sum():.6f}")

# Проверка уравнения (10): p̅ × (M - E) = 0
E1 = np.eye(len(states1))
check1 = p_bar1 @ (P1 - E1)
print("\nПроверка уравнения (10) p̅ × (M - E) = 0:")
print("p̅ × (M - E) =", check1)
print("Максимальная абсолютная величина:", np.max(np.abs(check1)))

# Обработка данных для студента 2
print("\n" + "=" * 70)
print("СТУДЕНТ 2")
print("=" * 70)

P2, states2 = build_transition_matrix(student2_transitions)
print(f"\nСостояния: {states2}")
print("\nМатрица переходных вероятностей P₂:")
print(P2)

# Решение уравнения (10)-(11) для студента 2
p_bar2 = solve_stationary_distribution(P2)
print("\nСтационарное распределение (решение уравнений (10)-(11)):")
for i, state in enumerate(states2):
    print(f"p̅₂({state}) = {p_bar2[i]:.6f}")

print(f"\nПроверка уравнения (11): Σⱼ p̅ⱼ = {p_bar2.sum():.6f}")

# Проверка уравнения (10): p̅ × (M - E) = 0
E2 = np.eye(len(states2))
check2 = p_bar2 @ (P2 - E2)
print("\nПроверка уравнения (10) p̅ × (M - E) = 0:")
print("p̅ × (M - E) =", check2)
print("Максимальная абсолютная величина:", np.max(np.abs(check2)))

# Сохранение результатов
results = {
    'student1': {
        'M': P1,  # M - матрица переходов
        'p_bar': p_bar1,  # p̅ - вектор стационарных вероятностей
        'states': states1
    },
    'student2': {
        'M': P2,
        'p_bar': p_bar2,
        'states': states2
    }
}

import pickle
with open('lab10_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 70)
print("Результаты сохранены в lab10_results.pkl")
print("=" * 70)
