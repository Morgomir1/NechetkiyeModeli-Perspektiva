# -*- coding: utf-8 -*-
"""
Лабораторная работа 10: Сравнение методов Мамдани и Сугено
Сравнение нечетких систем вывода для задачи управления
"""

import numpy as np
import pickle
import os

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def triangular_mf(x, a, b, c):
    """Треугольная функция принадлежности"""
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

def trapezoidal_mf(x, a, b, c, d):
    """Трапециевидная функция принадлежности"""
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))

def min_operator(x, y):
    """Оператор минимума для агрегации"""
    return np.minimum(x, y)

def max_operator(x, y):
    """Оператор максимума для агрегации"""
    return np.maximum(x, y)

def centroid_defuzzification(x, mf):
    """Дефаззификация методом центра тяжести"""
    if np.sum(mf) == 0:
        return 0
    return np.sum(x * mf) / np.sum(mf)

def mamdani_system(input1, input2):
    """Нечеткая система Мамдани"""
    # Определение функций принадлежности для входных переменных
    # Входная переменная 1: от 0 до 10
    low1 = triangular_mf(input1, 0, 0, 5)
    medium1 = triangular_mf(input1, 2, 5, 8)
    high1 = triangular_mf(input1, 5, 10, 10)
    
    # Входная переменная 2: от 0 до 10
    low2 = triangular_mf(input2, 0, 0, 5)
    medium2 = triangular_mf(input2, 2, 5, 8)
    high2 = triangular_mf(input2, 5, 10, 10)
    
    # Выходная переменная: от 0 до 20
    output_range = np.linspace(0, 20, 1000)
    
    # Функции принадлежности для выходной переменной
    def output_low(x):
        return triangular_mf(x, 0, 0, 10)
    
    def output_medium(x):
        return triangular_mf(x, 5, 10, 15)
    
    def output_high(x):
        return triangular_mf(x, 10, 20, 20)
    
    # Правила системы Мамдани
    rules = []
    
    # Правило 1: Если input1 низкое И input2 низкое, то output низкое
    rule1_strength = min_operator(low1, low2)
    rules.append((rule1_strength, output_low))
    
    # Правило 2: Если input1 низкое И input2 среднее, то output низкое
    rule2_strength = min_operator(low1, medium2)
    rules.append((rule2_strength, output_low))
    
    # Правило 3: Если input1 низкое И input2 высокое, то output среднее
    rule3_strength = min_operator(low1, high2)
    rules.append((rule3_strength, output_medium))
    
    # Правило 4: Если input1 среднее И input2 низкое, то output низкое
    rule4_strength = min_operator(medium1, low2)
    rules.append((rule4_strength, output_low))
    
    # Правило 5: Если input1 среднее И input2 среднее, то output среднее
    rule5_strength = min_operator(medium1, medium2)
    rules.append((rule5_strength, output_medium))
    
    # Правило 6: Если input1 среднее И input2 высокое, то output высокое
    rule6_strength = min_operator(medium1, high2)
    rules.append((rule6_strength, output_high))
    
    # Правило 7: Если input1 высокое И input2 низкое, то output среднее
    rule7_strength = min_operator(high1, low2)
    rules.append((rule7_strength, output_medium))
    
    # Правило 8: Если input1 высокое И input2 среднее, то output высокое
    rule8_strength = min_operator(high1, medium2)
    rules.append((rule8_strength, output_high))
    
    # Правило 9: Если input1 высокое И input2 высокое, то output высокое
    rule9_strength = min_operator(high1, high2)
    rules.append((rule9_strength, output_high))
    
    # Агрегация правил
    aggregated_mf = np.zeros_like(output_range)
    for strength, output_mf in rules:
        if strength > 0:
            output_values = output_mf(output_range)
            aggregated_mf = max_operator(aggregated_mf, min_operator(strength, output_values))
    
    # Дефаззификация
    output_value = centroid_defuzzification(output_range, aggregated_mf)
    
    return output_value, aggregated_mf, output_range

def sugeno_system(input1, input2):
    """Нечеткая система Сугено (первого порядка)"""
    # Определение функций принадлежности для входных переменных
    low1 = triangular_mf(input1, 0, 0, 5)
    medium1 = triangular_mf(input1, 2, 5, 8)
    high1 = triangular_mf(input1, 5, 10, 10)
    
    low2 = triangular_mf(input2, 0, 0, 5)
    medium2 = triangular_mf(input2, 2, 5, 8)
    high2 = triangular_mf(input2, 5, 10, 10)
    
    # Правила системы Сугено с линейными выходными функциями
    rules = []
    
    # Правило 1: Если input1 низкое И input2 низкое, то output = 2 + 0.3*input1 + 0.2*input2
    rule1_strength = min_operator(low1, low2)
    rule1_output = lambda: 2 + 0.3 * input1 + 0.2 * input2
    rules.append((rule1_strength, rule1_output))
    
    # Правило 2: Если input1 низкое И input2 среднее, то output = 3 + 0.2*input1 + 0.4*input2
    rule2_strength = min_operator(low1, medium2)
    rule2_output = lambda: 3 + 0.2 * input1 + 0.4 * input2
    rules.append((rule2_strength, rule2_output))
    
    # Правило 3: Если input1 низкое И input2 высокое, то output = 5 + 0.3*input1 + 0.5*input2
    rule3_strength = min_operator(low1, high2)
    rule3_output = lambda: 5 + 0.3 * input1 + 0.5 * input2
    rules.append((rule3_strength, rule3_output))
    
    # Правило 4: Если input1 среднее И input2 низкое, то output = 4 + 0.4*input1 + 0.3*input2
    rule4_strength = min_operator(medium1, low2)
    rule4_output = lambda: 4 + 0.4 * input1 + 0.3 * input2
    rules.append((rule4_strength, rule4_output))
    
    # Правило 5: Если input1 среднее И input2 среднее, то output = 8 + 0.5*input1 + 0.5*input2
    rule5_strength = min_operator(medium1, medium2)
    rule5_output = lambda: 8 + 0.5 * input1 + 0.5 * input2
    rules.append((rule5_strength, rule5_output))
    
    # Правило 6: Если input1 среднее И input2 высокое, то output = 12 + 0.6*input1 + 0.7*input2
    rule6_strength = min_operator(medium1, high2)
    rule6_output = lambda: 12 + 0.6 * input1 + 0.7 * input2
    rules.append((rule6_strength, rule6_output))
    
    # Правило 7: Если input1 высокое И input2 низкое, то output = 10 + 0.7*input1 + 0.4*input2
    rule7_strength = min_operator(high1, low2)
    rule7_output = lambda: 10 + 0.7 * input1 + 0.4 * input2
    rules.append((rule7_strength, rule7_output))
    
    # Правило 8: Если input1 высокое И input2 среднее, то output = 14 + 0.8*input1 + 0.6*input2
    rule8_strength = min_operator(high1, medium2)
    rule8_output = lambda: 14 + 0.8 * input1 + 0.6 * input2
    rules.append((rule8_strength, rule8_output))
    
    # Правило 9: Если input1 высокое И input2 высокое, то output = 18 + 0.9*input1 + 0.8*input2
    rule9_strength = min_operator(high1, high2)
    rule9_output = lambda: 18 + 0.9 * input1 + 0.8 * input2
    rules.append((rule9_strength, rule9_output))
    
    # Вычисление взвешенной суммы
    numerator = 0
    denominator = 0
    
    for strength, output_func in rules:
        if strength > 0:
            output_val = output_func()
            numerator += strength * output_val
            denominator += strength
    
    if denominator == 0:
        return 0
    
    output_value = numerator / denominator
    
    return output_value

# Тестирование систем
print("Тестирование нечетких систем...")

# Тестовые случаи
test_cases = [
    (1, 1),   # Низкие значения
    (2, 3),   # Низкие-средние
    (5, 5),   # Средние значения
    (7, 6),   # Средние-высокие
    (9, 9),   # Высокие значения
    (3, 8),   # Низкое-высокое
    (8, 2),   # Высокое-низкое
    (4, 7),   # Среднее-высокое
    (6, 4),   # Среднее-низкое
    (10, 10)  # Максимальные значения
]

results = {
    'mamdani': [],
    'sugeno': [],
    'comparison': []
}

for input1, input2 in test_cases:
    # Система Мамдани
    mamdani_output, mamdani_mf, mamdani_range = mamdani_system(input1, input2)
    
    # Система Сугено
    sugeno_output = sugeno_system(input1, input2)
    
    results['mamdani'].append({
        'input1': input1,
        'input2': input2,
        'output': mamdani_output,
        'membership': mamdani_mf,
        'range': mamdani_range
    })
    
    results['sugeno'].append({
        'input1': input1,
        'input2': input2,
        'output': sugeno_output
    })
    
    results['comparison'].append({
        'input1': input1,
        'input2': input2,
        'mamdani': mamdani_output,
        'sugeno': sugeno_output,
        'difference': abs(mamdani_output - sugeno_output)
    })
    
    print(f"Вход: ({input1:.1f}, {input2:.1f}) | Мамдани: {mamdani_output:.4f} | Сугено: {sugeno_output:.4f} | Разница: {abs(mamdani_output - sugeno_output):.4f}")

# Сохранение результатов
with open('fuzzy_comparison_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nРезультаты сохранены в fuzzy_comparison_results.pkl")
