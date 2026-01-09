import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches

# Настройка для русского языка
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10

# Задание 1: Отношение "Число x немного меньше числа y"
def task1_relation():
    """
    Задать на декартовом множестве {0, 1, 2, 3}× {0, 1, 2, 3, 4} 
    отношение: «Число x немного меньше числа y».
    """
    X = [0, 1, 2, 3]
    Y = [0, 1, 2, 3, 4]
    
    # Функция принадлежности для отношения "x немного меньше y"
    # Если x < y, то степень принадлежности зависит от разности
    # Если x >= y, то степень принадлежности = 0
    def mu_relation(x, y):
        if x >= y:
            return 0.0
        else:
            # Чем больше разность (y - x), тем выше степень принадлежности
            # Но если разность слишком большая, то это уже не "немного меньше"
            diff = y - x
            if diff == 1:
                return 1.0  # x немного меньше y
            elif diff == 2:
                return 0.7  # x меньше y, но не совсем "немного"
            elif diff == 3:
                return 0.4
            elif diff >= 4:
                return 0.1
            else:
                return 0.0
    
    # Создаем матрицу отношения
    relation_matrix = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            relation_matrix[i, j] = mu_relation(x, y)
    
    # Визуализация матрицы отношения
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(relation_matrix, cmap='gray', aspect='auto', vmin=0, vmax=1)
    
    # Устанавливаем метки
    ax.set_xticks(np.arange(len(Y)))
    ax.set_yticks(np.arange(len(X)))
    ax.set_xticklabels(Y)
    ax.set_yticklabels(X)
    
    # Добавляем значения в ячейки
    for i in range(len(X)):
        for j in range(len(Y)):
            text = ax.text(j, i, f'{relation_matrix[i, j]:.1f}',
                          ha="center", va="center", color="white" if relation_matrix[i, j] > 0.5 else "black")
    
    ax.set_xlabel('y', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title('Нечеткое отношение "Число x немного меньше числа y"', fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('task1_relation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return relation_matrix, X, Y

# Задание 2: Отношения нечетких множеств
def task2_fuzzy_sets():
    """
    Найти отношения двух нечетких множеств: 
    "Пить много кофе в день (чашек)" И "поздно ложиться"
    в дискретном и непрерывном виде.
    """
    # Дискретный случай
    # Множество "Пить много кофе" (чашек в день)
    coffee_discrete = {
        0: 0.0,
        1: 0.1,
        2: 0.3,
        3: 0.6,
        4: 0.8,
        5: 1.0,
        6: 1.0,
        7: 0.9,
        8: 0.7
    }
    
    # Множество "Поздно ложиться" (часы)
    late_discrete = {
        20: 0.0,
        21: 0.1,
        22: 0.3,
        23: 0.6,
        24: 0.9,
        1: 1.0,
        2: 1.0,
        3: 0.8,
        4: 0.5
    }
    
    # Пересечение (операция И) - берем минимум
    intersection_discrete = {}
    coffee_keys = list(coffee_discrete.keys())
    late_keys = list(late_discrete.keys())
    
    # Для визуализации создадим декартово произведение
    # Но для операции И нужно найти общее множество или использовать другой подход
    # Обычно операция И применяется к одному и тому же универсальному множеству
    # Здесь у нас разные множества, поэтому создадим отношение
    
    # Непрерывный случай
    def mu_coffee_continuous(x):
        """Функция принадлежности 'много кофе' (чашек в день)"""
        if x < 2:
            return 0.0
        elif 2 <= x <= 5:
            return (x - 2) / 3
        elif 5 <= x <= 6:
            return 1.0
        elif 6 < x <= 8:
            return 1.0 - 0.3 * (x - 6) / 2
        else:
            return 0.7
    
    def mu_late_continuous(x):
        """Функция принадлежности 'поздно ложиться' (часы, где 22 = 22:00)"""
        if x < 22:
            return 0.0
        elif 22 <= x <= 24:
            return (x - 22) / 2
        elif 24 < x <= 26:  # 24 = 0:00, 26 = 2:00
            return 1.0
        elif 26 < x <= 28:  # до 4:00
            return 1.0 - 0.5 * (x - 26) / 2
        else:
            return 0.5
    
    # Графики для дискретного случая
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Дискретное множество "много кофе"
    coffee_x = list(coffee_discrete.keys())
    coffee_y = list(coffee_discrete.values())
    axes[0, 0].bar(coffee_x, coffee_y, color='gray', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Количество чашек кофе в день', fontsize=11)
    axes[0, 0].set_ylabel('Степень принадлежности', fontsize=11)
    axes[0, 0].set_title('Дискретное множество "Пить много кофе"', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # Дискретное множество "поздно ложиться"
    late_x = list(late_discrete.keys())
    late_y = list(late_discrete.values())
    axes[0, 1].bar(late_x, late_y, color='gray', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Время отхода ко сну (часы)', fontsize=11)
    axes[0, 1].set_ylabel('Степень принадлежности', fontsize=11)
    axes[0, 1].set_title('Дискретное множество "Поздно ложиться"', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.1)
    
    # Непрерывное множество "много кофе"
    x_coffee = np.linspace(0, 10, 1000)
    y_coffee = [mu_coffee_continuous(x) for x in x_coffee]
    axes[1, 0].plot(x_coffee, y_coffee, 'k-', linewidth=2)
    axes[1, 0].fill_between(x_coffee, y_coffee, alpha=0.3, color='gray')
    axes[1, 0].set_xlabel('Количество чашек кофе в день', fontsize=11)
    axes[1, 0].set_ylabel('Степень принадлежности', fontsize=11)
    axes[1, 0].set_title('Непрерывное множество "Пить много кофе"', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.1)
    
    # Непрерывное множество "поздно ложиться"
    x_late = np.linspace(20, 30, 1000)
    y_late = [mu_late_continuous(x) for x in x_late]
    axes[1, 1].plot(x_late, y_late, 'k-', linewidth=2)
    axes[1, 1].fill_between(x_late, y_late, alpha=0.3, color='gray')
    axes[1, 1].set_xlabel('Время отхода ко сну (часы)', fontsize=11)
    axes[1, 1].set_ylabel('Степень принадлежности', fontsize=11)
    axes[1, 1].set_title('Непрерывное множество "Поздно ложиться"', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('task2_fuzzy_sets.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Декартово произведение и отношение
    # Создаем отношение на декартовом произведении
    coffee_values = np.linspace(0, 10, 50)
    late_values = np.linspace(20, 30, 50)
    
    # Матрица отношения (декартово произведение)
    relation_matrix = np.zeros((len(coffee_values), len(late_values)))
    for i, c in enumerate(coffee_values):
        for j, l in enumerate(late_values):
            # Используем операцию MIN для декартова произведения
            relation_matrix[i, j] = min(mu_coffee_continuous(c), mu_late_continuous(l))
    
    # Визуализация отношения
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(relation_matrix, cmap='gray', aspect='auto', origin='lower', 
                   extent=[late_values[0], late_values[-1], coffee_values[0], coffee_values[-1]])
    ax.set_xlabel('Время отхода ко сну (часы)', fontsize=12)
    ax.set_ylabel('Количество чашек кофе в день', fontsize=12)
    ax.set_title('Декартово произведение: "Много кофе" × "Поздно ложиться"', fontsize=14)
    plt.colorbar(im, ax=ax, label='Степень принадлежности')
    plt.tight_layout()
    plt.savefig('task2_relation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return coffee_discrete, late_discrete, mu_coffee_continuous, mu_late_continuous

# Задание 3: Задача о консалтинге в области выбора профессии
def task3_consulting():
    """
    Задача о консалтинге в области выбора профессии.
    Используем нечеткие отношения для определения подходящей профессии.
    """
    # Множество кандидатов
    candidates = ['Кандидат 1', 'Кандидат 2', 'Кандидат 3', 'Кандидат 4', 'Кандидат 5']
    
    # Множество профессий
    professions = ['Программист', 'Менеджер', 'Дизайнер', 'Аналитик', 'Маркетолог']
    
    # Характеристики кандидатов (нечеткие множества)
    # Математические способности
    math_skills = {
        'Кандидат 1': 0.9,
        'Кандидат 2': 0.6,
        'Кандидат 3': 0.4,
        'Кандидат 4': 0.8,
        'Кандидат 5': 0.5
    }
    
    # Творческие способности
    creative_skills = {
        'Кандидат 1': 0.3,
        'Кандидат 2': 0.7,
        'Кандидат 3': 0.9,
        'Кандидат 4': 0.5,
        'Кандидат 5': 0.8
    }
    
    # Коммуникативные способности
    comm_skills = {
        'Кандидат 1': 0.4,
        'Кандидат 2': 0.9,
        'Кандидат 3': 0.6,
        'Кандидат 4': 0.7,
        'Кандидат 5': 0.8
    }
    
    # Требования профессий к характеристикам
    # Программист: высокие математические, низкие творческие, средние коммуникативные
    # Менеджер: средние математические, низкие творческие, высокие коммуникативные
    # Дизайнер: низкие математические, высокие творческие, средние коммуникативные
    # Аналитик: высокие математические, низкие творческие, средние коммуникативные
    # Маркетолог: низкие математические, средние творческие, высокие коммуникативные
    
    profession_requirements = {
        'Программист': {'math': 0.9, 'creative': 0.2, 'comm': 0.5},
        'Менеджер': {'math': 0.5, 'creative': 0.3, 'comm': 0.9},
        'Дизайнер': {'math': 0.2, 'creative': 0.9, 'comm': 0.6},
        'Аналитик': {'math': 0.8, 'creative': 0.3, 'comm': 0.6},
        'Маркетолог': {'math': 0.3, 'creative': 0.7, 'comm': 0.9}
    }
    
    # Вычисляем совместимость кандидатов с профессиями
    # Используем операцию MIN для каждой характеристики, затем MAX для объединения
    compatibility_matrix = np.zeros((len(candidates), len(professions)))
    
    for i, candidate in enumerate(candidates):
        for j, profession in enumerate(professions):
            req = profession_requirements[profession]
            # Совместимость по каждой характеристике (MIN)
            math_comp = min(math_skills[candidate], req['math'])
            creative_comp = min(creative_skills[candidate], req['creative'])
            comm_comp = min(comm_skills[candidate], req['comm'])
            # Общая совместимость - используем среднее арифметическое для более гибкой оценки
            compatibility_matrix[i, j] = (math_comp + creative_comp + comm_comp) / 3
    
    # Визуализация матрицы совместимости
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(compatibility_matrix, cmap='gray', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(professions)))
    ax.set_yticks(np.arange(len(candidates)))
    ax.set_xticklabels(professions, rotation=45, ha='right')
    ax.set_yticklabels(candidates)
    
    # Добавляем значения в ячейки
    for i in range(len(candidates)):
        for j in range(len(professions)):
            text = ax.text(j, i, f'{compatibility_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if compatibility_matrix[i, j] > 0.5 else "black",
                          fontsize=9)
    
    ax.set_xlabel('Профессии', fontsize=12)
    ax.set_ylabel('Кандидаты', fontsize=12)
    ax.set_title('Матрица совместимости кандидатов с профессиями', fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Степень совместимости')
    plt.tight_layout()
    plt.savefig('task3_consulting.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Выводим матрицу для проверки
    print("\nМатрица совместимости:")
    print(f"{'Кандидат':<15}", end="")
    for prof in professions:
        print(f"{prof[:10]:>12}", end="")
    print()
    for i, candidate in enumerate(candidates):
        print(f"{candidate:<15}", end="")
        for j in range(len(professions)):
            print(f"{compatibility_matrix[i, j]:>12.2f}", end="")
        print()
    
    # Находим лучшие соответствия
    best_matches = {}
    for i, candidate in enumerate(candidates):
        best_prof_idx = np.argmax(compatibility_matrix[i, :])
        best_prof = professions[best_prof_idx]
        best_score = compatibility_matrix[i, best_prof_idx]
        best_matches[candidate] = (best_prof, best_score)
    
    return compatibility_matrix, candidates, professions, best_matches

if __name__ == '__main__':
    print("Генерация графиков для лабораторной работы 5...")
    
    print("\nЗадание 1: Отношение 'Число x немного меньше числа y'...")
    task1_relation()
    print("✓ График создан: task1_relation.png")
    
    print("\nЗадание 2: Отношения нечетких множеств...")
    task2_fuzzy_sets()
    print("✓ Графики созданы: task2_fuzzy_sets.png, task2_relation.png")
    
    print("\nЗадание 3: Задача о консалтинге...")
    task3_consulting()
    print("✓ График создан: task3_consulting.png")
    
    print("\nВсе графики успешно созданы!")

