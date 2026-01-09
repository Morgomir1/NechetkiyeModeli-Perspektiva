# -*- coding: utf-8 -*-
"""
Лабораторная работа: Нечеткая классификация
Выполнил: Тимошинов Егор Борисович, группа 16
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# Настройка для черно-белого вывода
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

# Обучающая выборка
# Наблюдение: 1    2    3    4    5    6    7
x1_values = [0.1, 0.4, 0.3, 0.5, 0.4, 0.2, 0.3]
x2_values = [0.7, 0.9, 0.6, 0.7, 0.5, 0.4, 0.1]
classes = [1, 1, 1, 2, 2, 2, 2]

# Новое наблюдение
new_observation = np.array([0.3, 0.8])

# Функции принадлежности для градаций S (малое), M (среднее), L (большое)
def mu_S(x):
    """Функция принадлежности для малого значения"""
    if x <= 0:
        return 1.0
    elif x >= 0.5:
        return 0.0
    else:
        return 1.0 - 2 * x

def mu_M(x):
    """Функция принадлежности для среднего значения"""
    if x <= 0 or x >= 1:
        return 0.0
    elif x <= 0.5:
        return 2 * x
    else:
        return 2 * (1 - x)

def mu_L(x):
    """Функция принадлежности для большого значения"""
    if x <= 0.5:
        return 0.0
    elif x >= 1:
        return 1.0
    else:
        return 2 * (x - 0.5)

# Вычисление принадлежности атрибутов к градациям для обучающей выборки
n_obs = len(x1_values)
membership_x1 = {'S': [], 'M': [], 'L': []}
membership_x2 = {'S': [], 'M': [], 'L': []}

for i in range(n_obs):
    membership_x1['S'].append(mu_S(x1_values[i]))
    membership_x1['M'].append(mu_M(x1_values[i]))
    membership_x1['L'].append(mu_L(x1_values[i]))
    
    membership_x2['S'].append(mu_S(x2_values[i]))
    membership_x2['M'].append(mu_M(x2_values[i]))
    membership_x2['L'].append(mu_L(x2_values[i]))

# Декартово произведение градаций
gradations = ['S', 'M', 'L']
cartesian_product = [(p1, p2) for p1 in gradations for p2 in gradations]

# Вычисление принадлежности к декартову произведению для обучающей выборки
mu_p_training = {}
for p in cartesian_product:
    p1, p2 = p
    mu_p_training[p] = []
    for i in range(n_obs):
        mu_p = membership_x1[p1][i] * membership_x2[p2][i]  # T-норма (произведение)
        mu_p_training[p].append(mu_p)

# Вычисление β для каждого класса
beta = {1: {}, 2: {}}
for p in cartesian_product:
    beta[1][p] = 0.0
    beta[2][p] = 0.0
    for i in range(n_obs):
        if classes[i] == 1:
            beta[1][p] += mu_p_training[p][i]
        else:
            beta[2][p] += mu_p_training[p][i]

# Вычисление степени доверия c_p
c_p = {}
for p in cartesian_product:
    beta1 = beta[1][p]
    beta2 = beta[2][p]
    if beta1 + beta2 == 0:
        c_p[p] = 0.0
    else:
        c_p[p] = abs(beta1 - beta2) / (beta1 + beta2)

# Определение классов для векторов p
P1 = []  # Векторы, характерные для класса 1
P2 = []  # Векторы, характерные для класса 2

for p in cartesian_product:
    if beta[1][p] > beta[2][p]:
        P1.append(p)
    elif beta[2][p] > beta[1][p]:
        P2.append(p)
    # Если равны, не добавляем ни в один класс

# Вычисление принадлежности нового наблюдения к градациям
new_mu_x1 = {'S': mu_S(new_observation[0]), 
             'M': mu_M(new_observation[0]), 
             'L': mu_L(new_observation[0])}
new_mu_x2 = {'S': mu_S(new_observation[1]), 
             'M': mu_M(new_observation[1]), 
             'L': mu_L(new_observation[1])}

# Вычисление принадлежности нового наблюдения к декартову произведению
new_mu_p = {}
for p in cartesian_product:
    p1, p2 = p
    new_mu_p[p] = new_mu_x1[p1] * new_mu_x2[p2]

# Вычисление функции принадлежности к классам
mu_class1 = 0.0
mu_class2 = 0.0

for p in P1:
    if c_p[p] > 0:  # Учитываем только векторы с ненулевой разделяющей способностью
        value = c_p[p] * new_mu_p[p]
        if value > mu_class1:
            mu_class1 = value

for p in P2:
    if c_p[p] > 0:
        value = c_p[p] * new_mu_p[p]
        if value > mu_class2:
            mu_class2 = value

# Определение класса
predicted_class = 1 if mu_class1 > mu_class2 else 2
max_membership = max(mu_class1, mu_class2)

# Вывод результатов в консоль
print("=" * 80)
print("НЕЧЕТКАЯ КЛАССИФИКАЦИЯ")
print("=" * 80)
print("\nОбучающая выборка:")
print("Наблюдение\tx1\tx2\tКласс")
for i in range(n_obs):
    print(f"{i+1}\t\t{x1_values[i]:.1f}\t{x2_values[i]:.1f}\t{classes[i]}")

print("\n" + "=" * 80)
print("Принадлежность атрибутов к градациям:")
print("\nПервый атрибут (x1):")
print("Наблюдение\tS\tM\tL")
for i in range(n_obs):
    print(f"{i+1}\t\t{membership_x1['S'][i]:.3f}\t{membership_x1['M'][i]:.3f}\t{membership_x1['L'][i]:.3f}")

print("\nВторой атрибут (x2):")
print("Наблюдение\tS\tM\tL")
for i in range(n_obs):
    print(f"{i+1}\t\t{membership_x2['S'][i]:.3f}\t{membership_x2['M'][i]:.3f}\t{membership_x2['L'][i]:.3f}")

print("\n" + "=" * 80)
print("Принадлежность к декартову произведению градаций:")
print("p=(p1,p2)", end="\t")
for i in range(n_obs):
    print(f"{i+1}", end="\t")
print()
for p in cartesian_product:
    print(f"{p}", end="\t")
    for i in range(n_obs):
        print(f"{mu_p_training[p][i]:.3f}", end="\t")
    print()

print("\n" + "=" * 80)
print("β для каждого класса:")
print("p=(p1,p2)", end="\t")
for i in range(n_obs):
    print(f"{i+1}", end="\t")
print("β¹\tβ²")
for p in cartesian_product:
    print(f"{p}", end="\t")
    for i in range(n_obs):
        print(f"{mu_p_training[p][i]:.3f}", end="\t")
    print(f"{beta[1][p]:.3f}\t{beta[2][p]:.3f}")

print("\n" + "=" * 80)
print("Степень доверия c_p и классы:")
print("p=(p1,p2)\tβ¹\tβ²\tc_p\tКласс")
for p in cartesian_product:
    cls = ""
    if p in P1:
        cls = "A1"
    elif p in P2:
        cls = "A2"
    print(f"{p}\t\t{beta[1][p]:.3f}\t{beta[2][p]:.3f}\t{c_p[p]:.3f}\t{cls}")

print("\n" + "=" * 80)
print("Векторы, характерные для классов:")
print(f"P1 (класс 1): {P1}")
print(f"P2 (класс 2): {P2}")

print("\n" + "=" * 80)
print("Новое наблюдение:", f"({new_observation[0]:.1f}, {new_observation[1]:.1f})")
print("\nПринадлежность к градациям:")
print(f"x1: S={new_mu_x1['S']:.3f}, M={new_mu_x1['M']:.3f}, L={new_mu_x1['L']:.3f}")
print(f"x2: S={new_mu_x2['S']:.3f}, M={new_mu_x2['M']:.3f}, L={new_mu_x2['L']:.3f}")

print("\nПринадлежность к декартову произведению:")
for p in cartesian_product:
    print(f"{p}: {new_mu_p[p]:.3f}")

print("\n" + "=" * 80)
print("Функция принадлежности к классам:")
print(f"μ_A1({new_observation[0]:.1f}, {new_observation[1]:.1f}) = {mu_class1:.6f}")
print(f"μ_A2({new_observation[0]:.1f}, {new_observation[1]:.1f}) = {mu_class2:.6f}")

print("\n" + "=" * 80)
print(f"РЕЗУЛЬТАТ КЛАССИФИКАЦИИ:")
print(f"Наблюдение ({new_observation[0]:.1f}, {new_observation[1]:.1f}) относится к КЛАССУ {predicted_class}")
print(f"Степень принадлежности: {max_membership:.6f}")
print("=" * 80)

# Функция для конвертации matplotlib figure в base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# Построение графика функций принадлежности
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 1, 1000)
mu_s = [mu_S(xi) for xi in x]
mu_m = [mu_M(xi) for xi in x]
mu_l = [mu_L(xi) for xi in x]

ax.plot(x, mu_s, 'k-', linewidth=2, label='S (малое)')
ax.plot(x, mu_m, 'k--', linewidth=2, label='M (среднее)')
ax.plot(x, mu_l, 'k:', linewidth=2, label='L (большое)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('μ(x)', fontsize=12)
ax.set_title('Функции принадлежности для градаций', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)

plt.tight_layout()
img_functions = fig_to_base64(fig)

# Отдельный график обучающей выборки
fig, ax = plt.subplots(figsize=(10, 8))
class1_x1 = [x1_values[i] for i in range(n_obs) if classes[i] == 1]
class1_x2 = [x2_values[i] for i in range(n_obs) if classes[i] == 1]
class2_x1 = [x1_values[i] for i in range(n_obs) if classes[i] == 2]
class2_x2 = [x2_values[i] for i in range(n_obs) if classes[i] == 2]

ax.scatter(class1_x1, class1_x2, c='white', marker='o', s=150, 
           edgecolors='black', linewidths=2, label='Класс 1')
ax.scatter(class2_x1, class2_x2, c='white', marker='s', s=150, 
           edgecolors='black', linewidths=2, label='Класс 2')
ax.scatter(new_observation[0], new_observation[1], c='white', marker='X', s=200, 
           edgecolors='black', linewidths=2, label='Новое наблюдение')
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_title('Обучающая выборка и новое наблюдение', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 0.6)
ax.set_ylim(0, 1.0)
plt.tight_layout()
img_data_points = fig_to_base64(fig)

# График степеней принадлежности к классам
fig, ax = plt.subplots(figsize=(8, 6))
classes_list = [1, 2]
membership_values = [mu_class1, mu_class2]
bars = ax.bar(classes_list, membership_values, color='white', 
              edgecolor='black', linewidth=2, width=0.5)
for bar, val in zip(bars, membership_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xlabel('Класс', fontsize=12)
ax.set_ylabel('Степень принадлежности', fontsize=12)
ax.set_title('Степени принадлежности нового наблюдения к классам', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(membership_values) * 1.2 if max(membership_values) > 0 else 1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
img_membership = fig_to_base64(fig)

# Сохранение результатов для HTML-отчета
results = {
    'training_data': {
        'x1': x1_values,
        'x2': x2_values,
        'classes': classes
    },
    'membership_x1': membership_x1,
    'membership_x2': membership_x2,
    'mu_p_training': mu_p_training,
    'beta': beta,
    'c_p': c_p,
    'P1': P1,
    'P2': P2,
    'new_observation': new_observation,
    'new_mu_x1': new_mu_x1,
    'new_mu_x2': new_mu_x2,
    'new_mu_p': new_mu_p,
    'mu_class1': mu_class1,
    'mu_class2': mu_class2,
    'predicted_class': predicted_class,
    'max_membership': max_membership,
    'img_functions': img_functions,
    'img_data_points': img_data_points,
    'img_membership': img_membership
}

import pickle
results_path = os.path.join(script_dir, 'fuzzy_classification_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\nРезультаты сохранены в файл {results_path}")
