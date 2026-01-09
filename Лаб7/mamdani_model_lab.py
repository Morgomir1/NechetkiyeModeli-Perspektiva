# -*- coding: utf-8 -*-
"""
Лабораторная работа 7 (часть 1): Модель Мамдани
Выполнил: Тимошинов Егор Борисович, группа 16
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Настройка для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# Определяем путь к директории скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ============================================================================
# ЗАДАЧА: Модель Мамдани для определения самочувствия на основе 
# количества выпитого кофе и времени отхода ко сну
# ============================================================================

# Входные переменные
# x - количество чашек кофе в день: [0, 6]
# y - время отхода ко сну: [20, 26] (20:00 - 2:00)

# Выходная переменная
# z - вероятность хорошего самочувствия: [0, 1]

# Дискретизация
x_disc = np.linspace(0, 6, 100)
y_disc = np.linspace(20, 26, 100)
z_disc = np.linspace(0, 1, 100)

# Функции принадлежности для входной переменной x (количество чашек кофе)
def mu_A1_malo(x):
    """Мало чашек кофе"""
    if x <= 2:
        return 1.0 - 0.5 * x
    else:
        return max(0.0, 1.0 - 0.5 * x)

def mu_A2_mnogo(x):
    """Много чашек кофе"""
    if x <= 2:
        return 0.5 * x
    else:
        return min(1.0, 0.5 * x)

# Функции принадлежности для входной переменной y (время отхода ко сну)
def mu_B1_rano(y):
    """Рано ложиться (до 22:00)"""
    if y <= 22:
        return 1.0 - 0.5 * (y - 20) / 2
    else:
        return max(0.0, 1.0 - 0.5 * (y - 20) / 2)

def mu_B2_pozdno(y):
    """Поздно ложиться (после 22:00)"""
    if y <= 22:
        return 0.5 * (y - 20) / 2
    else:
        return min(1.0, 0.5 * (y - 20) / 2)

# Функции принадлежности для выходной переменной z (самочувствие)
def mu_C1_horosho(z):
    """Чувствовать себя хорошо"""
    if z <= 0.4:
        return z / 0.4
    elif z <= 0.6:
        return 1.0
    elif z <= 0.8:
        return 1.0 - (z - 0.6) / 0.2
    else:
        return 0.0

def mu_C2_normalno(z):
    """Чувствовать себя нормально"""
    if z <= 0.2:
        return 0.0
    elif z <= 0.4:
        return (z - 0.2) / 0.2
    elif z <= 0.6:
        return 1.0
    elif z <= 0.8:
        return 1.0 - (z - 0.6) / 0.2
    else:
        return 0.0

def mu_C3_ploho(z):
    """Чувствовать себя плохо"""
    if z <= 0.2:
        return 1.0
    elif z <= 0.4:
        return 1.0 - (z - 0.2) / 0.2
    else:
        return 0.0

# Векторизованные функции
mu_A1_vec = np.vectorize(mu_A1_malo)
mu_A2_vec = np.vectorize(mu_A2_mnogo)
mu_B1_vec = np.vectorize(mu_B1_rano)
mu_B2_vec = np.vectorize(mu_B2_pozdno)
mu_C1_vec = np.vectorize(mu_C1_horosho)
mu_C2_vec = np.vectorize(mu_C2_normalno)
mu_C3_vec = np.vectorize(mu_C3_ploho)

# Система правил Мамдани
# R1: ЕСЛИ x = A1 (мало) И y = B1 (рано) ТО z = C1 (хорошо)
# R2: ЕСЛИ x = A1 (мало) И y = B2 (поздно) ТО z = C2 (нормально)
# R3: ЕСЛИ x = A2 (много) И y = B1 (рано) ТО z = C2 (нормально)
# R4: ЕСЛИ x = A2 (много) И y = B2 (поздно) ТО z = C3 (плохо)

print("=" * 80)
print("МОДЕЛЬ МАМДАНИ: Кофе и самочувствие")
print("=" * 80)

# Пример: входные значения
x_input = 1.5  # не очень много чашек кофе
y_input = 21.5  # не очень поздно

print(f"\nВходные значения:")
print(f"x (чашки кофе) = {x_input}")
print(f"y (время) = {y_input}")

# Шаг 1: ФАЗЗИФИКАЦИЯ
# Вычисляем степени принадлежности входных значений к нечетким множествам
mu_A1_input = mu_A1_malo(x_input)
mu_A2_input = mu_A2_mnogo(x_input)
mu_B1_input = mu_B1_rano(y_input)
mu_B2_input = mu_B2_pozdno(y_input)

print(f"\nФаззификация входных значений:")
print(f"μ_A1(мало)({x_input}) = {mu_A1_input:.4f}")
print(f"μ_A2(много)({x_input}) = {mu_A2_input:.4f}")
print(f"μ_B1(рано)({y_input}) = {mu_B1_input:.4f}")
print(f"μ_B2(поздно)({y_input}) = {mu_B2_input:.4f}")

# Шаг 2: ВЫЧИСЛЕНИЕ СТЕПЕНЕЙ АКТИВАЦИИ ПРАВИЛ (по методу Мамдани - MIN для условий)
h1 = min(mu_A1_input, mu_B1_input)  # R1: A1 И B1
h2 = min(mu_A1_input, mu_B2_input)  # R2: A1 И B2
h3 = min(mu_A2_input, mu_B1_input)  # R3: A2 И B1
h4 = min(mu_A2_input, mu_B2_input)  # R4: A2 И B2

print(f"\nСтепени активации правил (MIN для условий):")
print(f"h1 (R1: мало И рано) = min({mu_A1_input:.4f}, {mu_B1_input:.4f}) = {h1:.4f}")
print(f"h2 (R2: мало И поздно) = min({mu_A1_input:.4f}, {mu_B2_input:.4f}) = {h2:.4f}")
print(f"h3 (R3: много И рано) = min({mu_A2_input:.4f}, {mu_B1_input:.4f}) = {h3:.4f}")
print(f"h4 (R4: много И поздно) = min({mu_A2_input:.4f}, {mu_B2_input:.4f}) = {h4:.4f}")

# Шаг 3: ОГРАНИЧЕНИЕ ВЫХОДНЫХ МНОЖЕСТВ (усечение по h)
# Для каждого правила ограничиваем выходное множество значением h
# μ_C*(z) = min(h, μ_C(z))

print(f"\nАгрегация выходных множеств (MAX для объединения):")

# Вычисляем результирующую функцию принадлежности
mu_result = np.zeros_like(z_disc)
for i, z in enumerate(z_disc):
    # Для каждого правила: min(h, μ_C(z)), затем MAX всех правил
    mu_R1 = min(h1, mu_C1_horosho(z))  # R1 -> C1
    mu_R2 = min(h2, mu_C2_normalno(z))  # R2 -> C2
    mu_R3 = min(h3, mu_C2_normalno(z))  # R3 -> C2
    mu_R4 = min(h4, mu_C3_ploho(z))  # R4 -> C3
    
    # Агрегация: MAX
    mu_result[i] = max(mu_R1, mu_R2, mu_R3, mu_R4)

# Шаг 4: ДЕФАЗЗИФИКАЦИЯ
# Метод центра тяжести (центроид)
def defuzzify_centroid(z_values, mu_values):
    """Дефаззификация методом центра тяжести"""
    # Используем метод трапеций для численного интегрирования
    # Вычисляем интеграл от z * μ(z) и от μ(z)
    numerator = 0.0
    denominator = 0.0
    for i in range(len(z_values) - 1):
        dz = z_values[i + 1] - z_values[i]
        # Метод трапеций
        numerator += dz * (z_values[i] * mu_values[i] + z_values[i + 1] * mu_values[i + 1]) / 2.0
        denominator += dz * (mu_values[i] + mu_values[i + 1]) / 2.0
    if denominator == 0:
        return 0.0
    return numerator / denominator

z_output = defuzzify_centroid(z_disc, mu_result)

print(f"\nДефаззификация (центроид):")
print(f"z* = {z_output:.4f}")

# ============================================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================================

# График 1: Входные нечеткие множества для x
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Модель Мамдани: Фаззификация входных переменных', fontsize=14, fontweight='bold')

# Входная переменная x
axes[0, 0].plot(x_disc, mu_A1_vec(x_disc), color='blue', linewidth=2, label='A1: Мало')
axes[0, 0].plot(x_disc, mu_A2_vec(x_disc), color='red', linewidth=2, label='A2: Много')
axes[0, 0].axvline(x_input, color='green', linestyle='--', linewidth=2, label=f'Вход: x={x_input}')
axes[0, 0].set_xlabel('x (количество чашек кофе)', fontsize=11)
axes[0, 0].set_ylabel('μ(x)', fontsize=11)
axes[0, 0].set_title('Входная переменная: Количество чашек кофе', fontsize=12)
axes[0, 0].legend(loc='best')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 1.1)

# Входная переменная y
axes[0, 1].plot(y_disc, mu_B1_vec(y_disc), color='blue', linewidth=2, label='B1: Рано')
axes[0, 1].plot(y_disc, mu_B2_vec(y_disc), color='red', linewidth=2, label='B2: Поздно')
axes[0, 1].axvline(y_input, color='green', linestyle='--', linewidth=2, label=f'Вход: y={y_input}')
axes[0, 1].set_xlabel('y (время отхода ко сну)', fontsize=11)
axes[0, 1].set_ylabel('μ(y)', fontsize=11)
axes[0, 1].set_title('Входная переменная: Время отхода ко сну', fontsize=12)
axes[0, 1].legend(loc='best')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1.1)

# Выходные нечеткие множества
axes[1, 0].plot(z_disc, mu_C1_vec(z_disc), color='green', linewidth=2, label='C1: Хорошо')
axes[1, 0].plot(z_disc, mu_C2_vec(z_disc), color='orange', linewidth=2, label='C2: Нормально')
axes[1, 0].plot(z_disc, mu_C3_vec(z_disc), color='purple', linewidth=2, label='C3: Плохо')
axes[1, 0].set_xlabel('z (вероятность хорошего самочувствия)', fontsize=11)
axes[1, 0].set_ylabel('μ(z)', fontsize=11)
axes[1, 0].set_title('Выходные нечеткие множества', fontsize=12)
axes[1, 0].legend(loc='best')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 1.1)

# Результирующее выходное множество и дефаззификация
axes[1, 1].fill_between(z_disc, mu_result, 0, color='gray', alpha=0.3, label='Результирующее множество')
axes[1, 1].plot(z_disc, mu_result, color='black', linewidth=2)
axes[1, 1].axvline(z_output, color='red', linestyle='--', linewidth=2, 
                   label=f'Дефаззификация: z*={z_output:.4f}')
axes[1, 1].set_xlabel('z (вероятность хорошего самочувствия)', fontsize=11)
axes[1, 1].set_ylabel('μ(z)', fontsize=11)
axes[1, 1].set_title('Агрегация выходов правил и дефаззификация', fontsize=12)
axes[1, 1].legend(loc='best')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('mamdani_input_output.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Визуализация процесса вывода Мамдани
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Модель Мамдани: Процесс вывода по правилам', fontsize=14, fontweight='bold')

# Правило 1: R1 -> C1 (хорошо)
mu_R1_rule = np.array([min(h1, mu_C1_horosho(z)) for z in z_disc])
axes[0, 0].fill_between(z_disc, mu_R1_rule, 0, color='green', alpha=0.3)
axes[0, 0].plot(z_disc, mu_R1_rule, color='green', linewidth=2, label=f'R1: h1={h1:.4f}')
axes[0, 0].plot(z_disc, mu_C1_vec(z_disc), color='green', linestyle='--', linewidth=1, alpha=0.5, label='C1 исходное')
axes[0, 0].set_xlabel('z', fontsize=11)
axes[0, 0].set_ylabel('μ(z)', fontsize=11)
axes[0, 0].set_title('Правило R1: мало И рано → хорошо', fontsize=12)
axes[0, 0].legend(loc='best')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, 1.1)

# Правило 2: R2 -> C2 (нормально)
mu_R2_rule = np.array([min(h2, mu_C2_normalno(z)) for z in z_disc])
axes[0, 1].fill_between(z_disc, mu_R2_rule, 0, color='orange', alpha=0.3)
axes[0, 1].plot(z_disc, mu_R2_rule, color='orange', linewidth=2, label=f'R2: h2={h2:.4f}')
axes[0, 1].plot(z_disc, mu_C2_vec(z_disc), color='orange', linestyle='--', linewidth=1, alpha=0.5, label='C2 исходное')
axes[0, 1].set_xlabel('z', fontsize=11)
axes[0, 1].set_ylabel('μ(z)', fontsize=11)
axes[0, 1].set_title('Правило R2: мало И поздно → нормально', fontsize=12)
axes[0, 1].legend(loc='best')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1.1)

# Правило 3: R3 -> C2 (нормально)
mu_R3_rule = np.array([min(h3, mu_C2_normalno(z)) for z in z_disc])
axes[1, 0].fill_between(z_disc, mu_R3_rule, 0, color='orange', alpha=0.3)
axes[1, 0].plot(z_disc, mu_R3_rule, color='orange', linewidth=2, label=f'R3: h3={h3:.4f}')
axes[1, 0].plot(z_disc, mu_C2_vec(z_disc), color='orange', linestyle='--', linewidth=1, alpha=0.5, label='C2 исходное')
axes[1, 0].set_xlabel('z', fontsize=11)
axes[1, 0].set_ylabel('μ(z)', fontsize=11)
axes[1, 0].set_title('Правило R3: много И рано → нормально', fontsize=12)
axes[1, 0].legend(loc='best')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 1.1)

# Правило 4: R4 -> C3 (плохо)
mu_R4_rule = np.array([min(h4, mu_C3_ploho(z)) for z in z_disc])
axes[1, 1].fill_between(z_disc, mu_R4_rule, 0, color='purple', alpha=0.3)
axes[1, 1].plot(z_disc, mu_R4_rule, color='purple', linewidth=2, label=f'R4: h4={h4:.4f}')
axes[1, 1].plot(z_disc, mu_C3_vec(z_disc), color='purple', linestyle='--', linewidth=1, alpha=0.5, label='C3 исходное')
axes[1, 1].set_xlabel('z', fontsize=11)
axes[1, 1].set_ylabel('μ(z)', fontsize=11)
axes[1, 1].set_title('Правило R4: много И поздно → плохо', fontsize=12)
axes[1, 1].legend(loc='best')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('mamdani_rules.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: Агрегированное выходное множество
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(z_disc, mu_result, 0, color='gray', alpha=0.3, label='Агрегированное множество')
ax.plot(z_disc, mu_result, color='black', linewidth=2, label='μ_res(z)')
ax.axvline(z_output, color='red', linestyle='--', linewidth=2, 
           label=f'Центроид: z*={z_output:.4f}')
ax.set_xlabel('z (вероятность хорошего самочувствия)', fontsize=12)
ax.set_ylabel('μ(z)', fontsize=12)
ax.set_title('Агрегированное выходное множество и дефаззификация', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('mamdani_aggregation.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("Графики сохранены:")
print("- mamdani_input_output.png")
print("- mamdani_rules.png")
print("- mamdani_aggregation.png")
print("=" * 80)
