"""
Проверка правильности вычисления матрицы совместимости
"""
import numpy as np

# Характеристики кандидатов
math_skills = {
    'Кандидат 1': 0.9,
    'Кандидат 2': 0.6,
    'Кандидат 3': 0.4,
    'Кандидат 4': 0.8,
    'Кандидат 5': 0.5
}

creative_skills = {
    'Кандидат 1': 0.3,
    'Кандидат 2': 0.7,
    'Кандидат 3': 0.9,
    'Кандидат 4': 0.5,
    'Кандидат 5': 0.8
}

comm_skills = {
    'Кандидат 1': 0.4,
    'Кандидат 2': 0.9,
    'Кандидат 3': 0.6,
    'Кандидат 4': 0.7,
    'Кандидат 5': 0.8
}

# Требования профессий
profession_requirements = {
    'Программист': {'math': 0.9, 'creative': 0.2, 'comm': 0.5},
    'Менеджер': {'math': 0.5, 'creative': 0.3, 'comm': 0.9},
    'Дизайнер': {'math': 0.2, 'creative': 0.9, 'comm': 0.6},
    'Аналитик': {'math': 0.8, 'creative': 0.3, 'comm': 0.6},
    'Маркетолог': {'math': 0.3, 'creative': 0.7, 'comm': 0.9}
}

candidates = ['Кандидат 1', 'Кандидат 2', 'Кандидат 3', 'Кандидат 4', 'Кандидат 5']
professions = ['Программист', 'Менеджер', 'Дизайнер', 'Аналитик', 'Маркетолог']

print("=" * 80)
print("ПРОВЕРКА ВЫЧИСЛЕНИЯ СОВМЕСТИМОСТИ")
print("=" * 80)

# Вычисляем совместимость
compatibility_matrix = np.zeros((len(candidates), len(professions)))

for i, candidate in enumerate(candidates):
    for j, profession in enumerate(professions):
        req = profession_requirements[profession]
        # Совместимость по каждой характеристике (MIN)
        math_comp = min(math_skills[candidate], req['math'])
        creative_comp = min(creative_skills[candidate], req['creative'])
        comm_comp = min(comm_skills[candidate], req['comm'])
        # Общая совместимость (MIN всех трех)
        compatibility_matrix[i, j] = min(math_comp, creative_comp, comm_comp)
        
        # Выводим детали для первых нескольких комбинаций
        if i < 2 and j < 2:
            print(f"\n{candidate} - {profession}:")
            print(f"  Математические: min({math_skills[candidate]}, {req['math']}) = {math_comp}")
            print(f"  Творческие: min({creative_skills[candidate]}, {req['creative']}) = {creative_comp}")
            print(f"  Коммуникативные: min({comm_skills[candidate]}, {req['comm']}) = {comm_comp}")
            print(f"  Общая совместимость: min({math_comp}, {creative_comp}, {comm_comp}) = {compatibility_matrix[i, j]:.2f}")

print("\n" + "=" * 80)
print("МАТРИЦА СОВМЕСТИМОСТИ")
print("=" * 80)
print(f"{'Кандидат':<15}", end="")
for prof in professions:
    print(f"{prof[:10]:>12}", end="")
print()

for i, candidate in enumerate(candidates):
    print(f"{candidate:<15}", end="")
    for j in range(len(professions)):
        print(f"{compatibility_matrix[i, j]:>12.2f}", end="")
    print()

print("\n" + "=" * 80)
print("СРАВНЕНИЕ С ЗНАЧЕНИЯМИ В HTML")
print("=" * 80)

# Значения из HTML
html_values = {
    ('Кандидат 1', 'Программист'): 0.4,
    ('Кандидат 1', 'Менеджер'): 0.4,
    ('Кандидат 1', 'Дизайнер'): 0.2,
    ('Кандидат 1', 'Аналитик'): 0.4,
    ('Кандидат 1', 'Маркетолог'): 0.3,
    ('Кандидат 2', 'Программист'): 0.5,
    ('Кандидат 2', 'Менеджер'): 0.9,
    ('Кандидат 2', 'Дизайнер'): 0.6,
    ('Кандидат 2', 'Аналитик'): 0.5,
    ('Кандидат 2', 'Маркетолог'): 0.7,
}

print(f"{'Кандидат':<15} {'Профессия':<15} {'Вычислено':<12} {'В HTML':<12} {'Разница':<12}")
print("-" * 80)

for i, candidate in enumerate(candidates):
    for j, profession in enumerate(professions):
        computed = compatibility_matrix[i, j]
        html_val = html_values.get((candidate, profession), None)
        if html_val is not None:
            diff = abs(computed - html_val)
            match = "✓" if diff < 0.01 else "✗"
            print(f"{candidate:<15} {profession:<15} {computed:>12.2f} {html_val:>12.2f} {diff:>12.2f} {match}")

