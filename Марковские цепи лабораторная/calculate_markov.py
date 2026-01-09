import numpy as np
import matplotlib.pyplot as plt

# Задание 1: Техническое устройство (ТУ)
# Вариант 1: p12=0.1, p13=0.1, p14=0.1, p21=0.2, p24=0.2, p31=0.2, p34=0.3

# Состояния: S1 - исправно, S2 - требует наладки, S3 - требует ремонта, S4 - списано

# Матрица переходных вероятностей
P_task1 = np.array([
    [0.7, 0.1, 0.1, 0.1],  # S1: остается в S1 с вероятностью 0.7 (1-0.1-0.1-0.1)
    [0.2, 0.6, 0.0, 0.2],  # S2: остается в S2 с вероятностью 0.6 (1-0.2-0.2)
    [0.2, 0.0, 0.5, 0.3],  # S3: остается в S3 с вероятностью 0.5 (1-0.2-0.3)
    [0.0, 0.0, 0.0, 1.0]   # S4: поглощающее состояние
])

# Начальное состояние: P(0) = [1, 0, 0, 0] (система в состоянии S1)
P0_task1 = np.array([1.0, 0.0, 0.0, 0.0])

# Вычисление вероятностей для k=1, 2, 3
results_task1 = {}
results_task1[0] = P0_task1
for k in range(1, 4):
    results_task1[k] = results_task1[k-1] @ P_task1

print("Задание 1: Вероятности состояний ТУ")
print("=" * 50)
for k in range(0, 4):
    print(f"k={k}: P₁={results_task1[k][0]:.4f}, P₂={results_task1[k][1]:.4f}, "
          f"P₃={results_task1[k][2]:.4f}, P₄={results_task1[k][3]:.4f}")
print()

# Задание 2: Банк
# Состояния: S1 (3%), S2 (3.5%), S3 (4%), S4 (4.2%)
# В конце предшествующего квартала процентная ставка составляла 4% (S3)
# Нужно найти вероятности в конце квартала (3 месяца)
# По описанию графа: возможны переходы между соседними состояниями и остаток в текущем
# Создадим типичную матрицу переходов для такого процесса
P_task2 = np.array([
    [0.5, 0.4, 0.1, 0.0],  # S1
    [0.3, 0.4, 0.3, 0.0],  # S2
    [0.0, 0.3, 0.5, 0.2],  # S3
    [0.0, 0.0, 0.4, 0.6]   # S4
])

# Начальное состояние: P(0) = [0, 0, 1, 0] (S3 - 4%)
P0_task2 = np.array([0.0, 0.0, 1.0, 0.0])

# Квартал = 3 месяца, вычисляем P(3) = P(0) * P^3
P3_task2 = P0_task2 @ np.linalg.matrix_power(P_task2, 3)

print("Задание 2: Вероятности состояний банка в конце квартала")
print("=" * 50)
print(f"P₁(3)={P3_task2[0]:.4f}, P₂(3)={P3_task2[1]:.4f}, "
      f"P₃(3)={P3_task2[2]:.4f}, P₄(3)={P3_task2[3]:.4f}")
print()

# Задание 3: Динамика изменения доходов
# Вариант 1: p̄ = (0.1, 0.9), P = [[0.5, 0.5], [0.4, 0.6]], R = [[9, 3], [3, -7]]

# Вектор предельных вероятностей
p_lim = np.array([0.1, 0.9])

# Матрица переходных вероятностей
P_task3 = np.array([
    [0.5, 0.5],
    [0.4, 0.6]
])

# Матрица доходов
R_task3 = np.array([
    [9, 3],
    [3, -7]
])

# Начальное состояние определяется из вектора предельных вероятностей
P0_task3 = p_lim.copy()

# Вычисление вероятностей и доходов для k=1, 2, 3
results_task3 = {}
income_task3 = {}
results_task3[0] = P0_task3

# Вычисляем доход за один переход
# Доход рассчитывается как сумма по всем переходам: G(k) = Σᵢ Σⱼ pᵢ(k) * pᵢⱼ * rᵢⱼ
def calculate_income(P, P_trans, R):
    """Вычисляет ожидаемый доход за один переход"""
    income = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            income += P[i] * P_trans[i, j] * R[i, j]
    return income

income_task3[0] = calculate_income(P0_task3, P_task3, R_task3)

for k in range(1, 4):
    results_task3[k] = results_task3[k-1] @ P_task3
    income_task3[k] = calculate_income(results_task3[k], P_task3, R_task3)

print("Задание 3: Динамика изменения доходов")
print("=" * 50)
print(f"Начальное состояние: P₁={results_task3[0][0]:.4f}, P₂={results_task3[0][1]:.4f}")
print(f"Доход G(0)={income_task3[0]:.2f}")
for k in range(1, 4):
    print(f"k={k}: P₁={results_task3[k][0]:.4f}, P₂={results_task3[k][1]:.4f}, "
          f"G({k})={income_task3[k]:.2f}")
print()

# Сохранение результатов для использования в HTML
np.savez('markov_results.npz',
         task1_results=results_task1,
         task2_result=P3_task2,
         task3_results=results_task3,
         task3_income=income_task3)

# Построение графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# График для задания 1
fig1, ax1 = plt.subplots(figsize=(10, 6))
k_values = list(results_task1.keys())
states = ['S1', 'S2', 'S3', 'S4']
colors = ['blue', 'green', 'orange', 'red']

for i, (state, color) in enumerate(zip(states, colors)):
    values = [results_task1[k][i] for k in k_values]
    ax1.plot(k_values, values, marker='o', linewidth=2, label=state, color=color)

ax1.set_xlabel('Шаг k', fontsize=12)
ax1.set_ylabel('Вероятность', fontsize=12)
ax1.set_title('Задание 1: Динамика вероятностей состояний ТУ', fontsize=14)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xticks(k_values)
plt.tight_layout()
plt.savefig('task1_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# График для задания 3
fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(14, 6))

# График вероятностей
k_values3 = list(results_task3.keys())
for i, (state, color) in enumerate(zip(['S1', 'S2'], ['blue', 'green'])):
    values = [results_task3[k][i] for k in k_values3]
    ax31.plot(k_values3, values, marker='o', linewidth=2, label=state, color=color)

ax31.set_xlabel('Переход k', fontsize=12)
ax31.set_ylabel('Вероятность', fontsize=12)
ax31.set_title('Динамика вероятностей состояний', fontsize=14)
ax31.legend()
ax31.grid(True, linestyle='--', alpha=0.3)
ax31.set_xticks(k_values3)

# График доходов
income_values = [income_task3[k] for k in k_values3]
ax32.plot(k_values3, income_values, marker='s', linewidth=2, color='red')
ax32.set_xlabel('Переход k', fontsize=12)
ax32.set_ylabel('Доход G(k)', fontsize=12)
ax32.set_title('Динамика изменения доходов', fontsize=14)
ax32.grid(True, linestyle='--', alpha=0.3)
ax32.set_xticks(k_values3)

plt.tight_layout()
plt.savefig('task3_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("Графики сохранены: task1_plot.png, task3_plot.png")

