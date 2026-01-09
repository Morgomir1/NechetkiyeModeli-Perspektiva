# -*- coding: utf-8 -*-
"""
Генерация графиков для лабораторной работы 10
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import base64
import io

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Загрузка результатов
with open('fuzzy_comparison_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def image_to_base64(fig):
    """Конвертирует matplotlib figure в base64 строку"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# 1. График сравнения выходных значений
fig1, ax1 = plt.subplots(figsize=(10, 6))
test_indices = range(len(results['comparison']))
mamdani_outputs = [r['mamdani'] for r in results['comparison']]
sugeno_outputs = [r['sugeno'] for r in results['comparison']]

ax1.plot(test_indices, mamdani_outputs, 'o-', label='Мамдани', linewidth=2, markersize=8, color='#1f77b4')
ax1.plot(test_indices, sugeno_outputs, 's-', label='Сугено', linewidth=2, markersize=8, color='#ff7f0e')
ax1.set_xlabel('Номер тестового случая', fontsize=12)
ax1.set_ylabel('Выходное значение', fontsize=12)
ax1.set_title('Сравнение выходных значений систем Мамдани и Сугено', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(test_indices)
plt.tight_layout()
comparison_plot = image_to_base64(fig1)

# 2. График разницы между методами
fig2, ax2 = plt.subplots(figsize=(10, 6))
differences = [r['difference'] for r in results['comparison']]
ax2.bar(test_indices, differences, color='#2ca02c', alpha=0.7)
ax2.set_xlabel('Номер тестового случая', fontsize=12)
ax2.set_ylabel('Абсолютная разница', fontsize=12)
ax2.set_title('Разница между выходными значениями методов', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(test_indices)
plt.tight_layout()
difference_plot = image_to_base64(fig2)

# 3. 3D поверхность для системы Мамдани
from mpl_toolkits.mplot3d import Axes3D

input1_range = np.linspace(0, 10, 20)
input2_range = np.linspace(0, 10, 20)
X, Y = np.meshgrid(input1_range, input2_range)
Z_mamdani = np.zeros_like(X)

# Импортируем функцию из основного скрипта
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzzy_comparison_lab import mamdani_system

for i in range(len(input1_range)):
    for j in range(len(input2_range)):
        output, _, _ = mamdani_system(input1_range[i], input2_range[j])
        Z_mamdani[j, i] = output

fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')
surf1 = ax3.plot_surface(X, Y, Z_mamdani, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
ax3.set_xlabel('Входная переменная 1', fontsize=11)
ax3.set_ylabel('Входная переменная 2', fontsize=11)
ax3.set_zlabel('Выходное значение', fontsize=11)
ax3.set_title('Поверхность выхода системы Мамдани', fontsize=14, fontweight='bold')
fig3.colorbar(surf1, ax=ax3, shrink=0.5)
mamdani_surface = image_to_base64(fig3)

# 4. 3D поверхность для системы Сугено
from fuzzy_comparison_lab import sugeno_system

Z_sugeno = np.zeros_like(X)
for i in range(len(input1_range)):
    for j in range(len(input2_range)):
        output = sugeno_system(input1_range[i], input2_range[j])
        Z_sugeno[j, i] = output

fig4 = plt.figure(figsize=(10, 8))
ax4 = fig4.add_subplot(111, projection='3d')
surf2 = ax4.plot_surface(X, Y, Z_sugeno, cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
ax4.set_xlabel('Входная переменная 1', fontsize=11)
ax4.set_ylabel('Входная переменная 2', fontsize=11)
ax4.set_zlabel('Выходное значение', fontsize=11)
ax4.set_title('Поверхность выхода системы Сугено', fontsize=14, fontweight='bold')
fig4.colorbar(surf2, ax=ax4, shrink=0.5)
sugeno_surface = image_to_base64(fig4)

# 5. График функций принадлежности входных переменных
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 5))

x_range = np.linspace(0, 10, 1000)

def triangular_mf(x, a, b, c):
    """Треугольная функция принадлежности с обработкой граничных случаев"""
    # Обработка случаев, когда a == b или b == c
    if a == b:
        # Левая сторона начинается сразу
        left = np.where(x <= a, 1.0, 0.0)
    else:
        left = (x - a) / (b - a)
    
    if b == c:
        # Правая сторона заканчивается сразу
        right = np.where(x >= c, 1.0, 0.0)
    else:
        right = (c - x) / (c - b)
    
    return np.maximum(0, np.minimum(left, right))

low = triangular_mf(x_range, 0, 0, 5)
medium = triangular_mf(x_range, 2, 5, 8)
high = triangular_mf(x_range, 5, 10, 10)

ax5a.plot(x_range, low, 'b-', label='Низкое', linewidth=2)
ax5a.plot(x_range, medium, 'g-', label='Среднее', linewidth=2)
ax5a.plot(x_range, high, 'r-', label='Высокое', linewidth=2)
ax5a.set_xlabel('Значение переменной', fontsize=11)
ax5a.set_ylabel('Степень принадлежности', fontsize=11)
ax5a.set_title('Функции принадлежности входной переменной 1', fontsize=12, fontweight='bold')
ax5a.legend(fontsize=10)
ax5a.grid(True, alpha=0.3)

ax5b.plot(x_range, low, 'b-', label='Низкое', linewidth=2)
ax5b.plot(x_range, medium, 'g-', label='Среднее', linewidth=2)
ax5b.plot(x_range, high, 'r-', label='Высокое', linewidth=2)
ax5b.set_xlabel('Значение переменной', fontsize=11)
ax5b.set_ylabel('Степень принадлежности', fontsize=11)
ax5b.set_title('Функции принадлежности входной переменной 2', fontsize=12, fontweight='bold')
ax5b.legend(fontsize=10)
ax5b.grid(True, alpha=0.3)

plt.tight_layout()
membership_plot = image_to_base64(fig5)

# Сохранение изображений
images = {
    'comparison_plot': comparison_plot,
    'difference_plot': difference_plot,
    'mamdani_surface': mamdani_surface,
    'sugeno_surface': sugeno_surface,
    'membership_plot': membership_plot
}

with open('plot_images.pkl', 'wb') as f:
    pickle.dump(images, f)

print("Графики успешно сгенерированы и сохранены")
