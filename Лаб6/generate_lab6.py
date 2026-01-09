import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка для русского языка
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10

# Задание 1: Дискретизация нечетких множеств с применением PROD и метода 2
def task1_discretization():
    """
    Дискретизация нечетких множеств с применением PROD (произведение) и метода 2.
    Метод 2: использование операции умножения вместо минимума для T-нормы.
    """
    # Определим нечеткие множества
    # A1 - мало чашек кофе
    # A2 - много чашек кофе
    # B1 - рано ложиться
    # B2 - поздно ложиться
    
    # Дискретизация для переменной x (чашки кофе): 0, 1, 2, 3, 4, 5, 6
    x_values = np.array([0, 1, 2, 3, 4, 5, 6])
    
    # Функции принадлежности для A1 (мало) и A2 (много)
    def mu_A1(x):
        """Мало чашек кофе"""
        if x <= 2:
            return 1.0 - 0.5 * x
        else:
            return max(0, 1.0 - 0.5 * x)
    
    def mu_A2(x):
        """Много чашек кофе"""
        if x <= 2:
            return 0.5 * x
        else:
            return min(1.0, 0.5 * x)
    
    # Дискретизация для переменной y (время отхода ко сну): 20, 21, 22, 23, 24, 1, 2
    # Представим как: 20, 21, 22, 23, 24, 25, 26 (где 25 = 1:00, 26 = 2:00)
    y_values = np.array([20, 21, 22, 23, 24, 25, 26])
    
    # Функции принадлежности для B1 (рано) и B2 (поздно)
    def mu_B1(y):
        """Рано ложиться"""
        if y <= 22:
            return 1.0 - 0.5 * (y - 20) / 2
        else:
            return max(0, 1.0 - 0.5 * (y - 20) / 2)
    
    def mu_B2(y):
        """Поздно ложиться"""
        if y <= 22:
            return 0.5 * (y - 20) / 2
        else:
            return min(1.0, 0.5 * (y - 20) / 2)
    
    # Вычисляем степени принадлежности для дискретных значений
    A1_values = np.array([mu_A1(x) for x in x_values])
    A2_values = np.array([mu_A2(x) for x in x_values])
    B1_values = np.array([mu_B1(y) for y in y_values])
    B2_values = np.array([mu_B2(y) for y in y_values])
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График A1 и A2
    axes[0, 0].plot(x_values, A1_values, 'ko-', linewidth=2, markersize=8, label='A1 (мало)')
    axes[0, 0].plot(x_values, A2_values, 'ks--', linewidth=2, markersize=8, label='A2 (много)')
    axes[0, 0].set_xlabel('Количество чашек кофе', fontsize=11)
    axes[0, 0].set_ylabel('Степень принадлежности', fontsize=11)
    axes[0, 0].set_title('Нечеткие множества для переменной x', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # График B1 и B2
    axes[0, 1].plot(y_values, B1_values, 'ko-', linewidth=2, markersize=8, label='B1 (рано)')
    axes[0, 1].plot(y_values, B2_values, 'ks--', linewidth=2, markersize=8, label='B2 (поздно)')
    axes[0, 1].set_xlabel('Время отхода ко сну (часы)', fontsize=11)
    axes[0, 1].set_ylabel('Степень принадлежности', fontsize=11)
    axes[0, 1].set_title('Нечеткие множества для переменной y', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.1)
    
    # Демонстрация операции PROD (произведение) вместо MIN
    # Создаем декартово произведение с использованием PROD
    prod_matrix = np.zeros((len(x_values), len(y_values)))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            # PROD: умножение вместо минимума
            prod_matrix[i, j] = mu_A1(x) * mu_B1(y)
    
    # Визуализация матрицы PROD
    im1 = axes[1, 0].imshow(prod_matrix, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_xticks(np.arange(len(y_values)))
    axes[1, 0].set_yticks(np.arange(len(x_values)))
    axes[1, 0].set_xticklabels(y_values)
    axes[1, 0].set_yticklabels(x_values)
    axes[1, 0].set_xlabel('y (время)', fontsize=11)
    axes[1, 0].set_ylabel('x (чашки кофе)', fontsize=11)
    axes[1, 0].set_title('Декартово произведение A1 × B1 (PROD)', fontsize=12)
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Сравнение MIN и PROD для одной пары значений
    min_values = []
    prod_values = []
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            min_val = min(mu_A1(x), mu_B1(y))
            prod_val = mu_A1(x) * mu_B1(y)
            min_values.append(min_val)
            prod_values.append(prod_val)
    
    axes[1, 1].plot(min_values, 'ko-', linewidth=2, markersize=6, label='MIN')
    axes[1, 1].plot(prod_values, 'ks--', linewidth=2, markersize=6, label='PROD')
    axes[1, 1].set_xlabel('Индекс пары (x, y)', fontsize=11)
    axes[1, 1].set_ylabel('Значение операции', fontsize=11)
    axes[1, 1].set_title('Сравнение MIN и PROD', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1_discretization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return x_values, y_values, A1_values, A2_values, B1_values, B2_values

# Задание 3: Задача про кофе и самочувствие
def task3_coffee_wellbeing():
    """
    Решение задачи про кофе и самочувствие двумя способами:
    1. Композиционное правило вывода
    2. Упрощенный метод вывода
    Также рассмотрим Max-Prod вместо Max-Min
    """
    # Определим нечеткие множества
    # Входные переменные
    # x - число выпитых чашек кофе: 0, 1, 2, 3, 4, 5, 6
    x_values = np.array([0, 1, 2, 3, 4, 5, 6])
    
    # y - время отхода ко сну: 20, 21, 22, 23, 24, 25, 26
    y_values = np.array([20, 21, 22, 23, 24, 25, 26])
    
    # Выходная переменная z - вероятность чувствовать себя нормально: 0, 0.2, 0.4, 0.6, 0.8, 1.0
    z_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Функции принадлежности для входных переменных
    def mu_A1(x):  # Мало чашек
        if x <= 2:
            return max(0, 1.0 - 0.5 * x)
        else:
            return 0.0
    
    def mu_A2(x):  # Много чашек
        if x <= 2:
            return 0.0
        elif 2 < x <= 4:
            return 0.5 * (x - 2)
        else:
            return 1.0
    
    def mu_B1(y):  # Рано
        if y <= 22:
            return max(0, 1.0 - 0.5 * (y - 20) / 2)
        else:
            return 0.0
    
    def mu_B2(y):  # Поздно
        if y <= 22:
            return 0.0
        elif 22 < y <= 24:
            return 0.5 * (y - 22) / 2
        else:
            return 1.0
    
    # Функции принадлежности для выходных переменных
    def mu_C1(z):  # Хорошо
        if z >= 0.8:
            return 1.0
        elif 0.6 <= z < 0.8:
            return (z - 0.6) / 0.2
        else:
            return 0.0
    
    def mu_C2(z):  # Нормально
        if 0.4 <= z <= 0.6:
            return 1.0
        elif 0.2 <= z < 0.4:
            return (z - 0.2) / 0.2
        elif 0.6 < z <= 0.8:
            return (0.8 - z) / 0.2
        else:
            return 0.0
    
    def mu_C3(z):  # Плохо
        if z <= 0.2:
            return 1.0
        elif 0.2 < z <= 0.4:
            return (0.4 - z) / 0.2
        else:
            return 0.0
    
    # Входные нечеткие множества
    # A* - не очень много чашек
    def mu_A_star(x):
        if x <= 2:
            return 1.0
        elif 2 < x <= 4:
            return 1.0 - 0.5 * (x - 2) / 2
        else:
            return 0.5
    
    # B* - не очень поздно
    def mu_B_star(y):
        if y <= 22:
            return 1.0
        elif 22 < y <= 24:
            return 1.0 - 0.5 * (y - 22) / 2
        else:
            return 0.5
    
    # Система правил
    # R1: ЕСЛИ x=A1 И y=B1 ТО z=C1 (мало кофе и рано = хорошо)
    # R2: ЕСЛИ x=A1 И y=B2 ТО z=C2 (мало кофе и поздно = нормально)
    # R3: ЕСЛИ x=A2 И y=B1 ТО z=C2 (много кофе и рано = нормально)
    # R4: ЕСЛИ x=A2 И y=B2 ТО z=C3 (много кофе и поздно = плохо)
    
    # Вычисляем степени выполнения условий правил
    # Для входов A* и B*
    A_star_values = np.array([mu_A_star(x) for x in x_values])
    B_star_values = np.array([mu_B_star(y) for y in y_values])
    
    # Степени выполнения условий (используя MIN для "И")
    # h = max_x min(μ_A(x), μ_A*(x)) для каждого условия
    h_A1 = max([min(mu_A1(x), mu_A_star(x)) for x in x_values])
    h_A2 = max([min(mu_A2(x), mu_A_star(x)) for x in x_values])
    h_B1 = max([min(mu_B1(y), mu_B_star(y)) for y in y_values])
    h_B2 = max([min(mu_B2(y), mu_B_star(y)) for y in y_values])
    
    # Для правила с двумя условиями: h = min(h_A, h_B)
    h1 = min(h_A1, h_B1)  # R1: A1 И B1
    h2 = min(h_A1, h_B2)  # R2: A1 И B2
    h3 = min(h_A2, h_B1)  # R3: A2 И B1
    h4 = min(h_A2, h_B2)  # R4: A2 И B2
    
    print(f"Степени выполнения условий правил (Max-Min):")
    print(f"h1 = {h1:.3f}, h2 = {h2:.3f}, h3 = {h3:.3f}, h4 = {h4:.3f}")
    
    # Упрощенный метод вывода (Max-Min)
    # Модифицированные функции принадлежности заключений
    mu_res_min = np.zeros(len(z_values))
    for i, z in enumerate(z_values):
        mu_res_min[i] = max(
            min(h1, mu_C1(z)),
            min(h2, mu_C2(z)),
            min(h3, mu_C2(z)),
            min(h4, mu_C3(z))
        )
    
    # Max-Prod метод
    # Степени выполнения условий правил (используя PROD для "И")
    h_A1_prod = max([mu_A1(x) * mu_A_star(x) for x in x_values])
    h_A2_prod = max([mu_A2(x) * mu_A_star(x) for x in x_values])
    h_B1_prod = max([mu_B1(y) * mu_B_star(y) for y in y_values])
    h_B2_prod = max([mu_B2(y) * mu_B_star(y) for y in y_values])
    
    # Для правила с двумя условиями: h = h_A * h_B (PROD вместо MIN)
    h1_prod = h_A1_prod * h_B1_prod  # R1: A1 И B1
    h2_prod = h_A1_prod * h_B2_prod  # R2: A1 И B2
    h3_prod = h_A2_prod * h_B1_prod  # R3: A2 И B1
    h4_prod = h_A2_prod * h_B2_prod  # R4: A2 И B2
    
    print(f"\nСтепени выполнения условий правил (Max-Prod):")
    print(f"h1 = {h1_prod:.3f}, h2 = {h2_prod:.3f}, h3 = {h3_prod:.3f}, h4 = {h4_prod:.3f}")
    
    # Модифицированные функции принадлежности заключений (Max-Prod)
    mu_res_prod = np.zeros(len(z_values))
    for i, z in enumerate(z_values):
        mu_res_prod[i] = max(
            h1_prod * mu_C1(z),
            h2_prod * mu_C2(z),
            h3_prod * mu_C2(z),
            h4_prod * mu_C3(z)
        )
    
    # Сохраняем результаты в файл для использования в HTML
    with open('task3_results.txt', 'w', encoding='utf-8') as f:
        f.write("Результаты решения задачи о кофе и самочувствии\n")
        f.write("=" * 50 + "\n\n")
        f.write("Степени выполнения условий правил (Max-Min):\n")
        f.write(f"h1 = {h1:.3f}\n")
        f.write(f"h2 = {h2:.3f}\n")
        f.write(f"h3 = {h3:.3f}\n")
        f.write(f"h4 = {h4:.3f}\n\n")
        f.write("Степени выполнения условий правил (Max-Prod):\n")
        f.write(f"h1 = {h1_prod:.3f}\n")
        f.write(f"h2 = {h2_prod:.3f}\n")
        f.write(f"h3 = {h3_prod:.3f}\n")
        f.write(f"h4 = {h4_prod:.3f}\n\n")
        f.write("Результирующая функция принадлежности (Max-Min):\n")
        for i, z in enumerate(z_values):
            f.write(f"μ({z:.1f}) = {mu_res_min[i]:.3f}\n")
        f.write("\nРезультирующая функция принадлежности (Max-Prod):\n")
        for i, z in enumerate(z_values):
            f.write(f"μ({z:.1f}) = {mu_res_prod[i]:.3f}\n")
    
    # Композиционное правило вывода
    # Создаем нечеткие отношения для каждого правила
    # R1: A1 × B1 → C1
    R1 = np.zeros((len(x_values), len(y_values), len(z_values)))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            for k, z in enumerate(z_values):
                # Импликация: min(mu_A1(x), mu_B1(y)) → mu_C1(z)
                # Используем min для импликации
                R1[i, j, k] = min(min(mu_A1(x), mu_B1(y)), mu_C1(z))
    
    # Аналогично для других правил
    R2 = np.zeros((len(x_values), len(y_values), len(z_values)))
    R3 = np.zeros((len(x_values), len(y_values), len(z_values)))
    R4 = np.zeros((len(x_values), len(y_values), len(z_values)))
    
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            for k, z in enumerate(z_values):
                R2[i, j, k] = min(min(mu_A1(x), mu_B2(y)), mu_C2(z))
                R3[i, j, k] = min(min(mu_A2(x), mu_B1(y)), mu_C2(z))
                R4[i, j, k] = min(min(mu_A2(x), mu_B2(y)), mu_C3(z))
    
    # Объединяем правила (MAX)
    R = np.maximum(np.maximum(R1, R2), np.maximum(R3, R4))
    
    # Цилиндрическое продолжение A* × B* на X × Y × Z
    A_star_cyl = np.zeros((len(x_values), len(y_values), len(z_values)))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            for k, z in enumerate(z_values):
                A_star_cyl[i, j, k] = min(mu_A_star(x), mu_B_star(y))
    
    # Композиция: min(A*_cyl, R), затем проекция на Z
    composition = np.minimum(A_star_cyl, R)
    mu_res_composition = np.max(composition, axis=(0, 1))
    
    # Визуализация результатов
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График входных нечетких множеств A*
    axes[0, 0].plot(x_values, A_star_values, 'ko-', linewidth=2, markersize=8, label='A* (не очень много)')
    axes[0, 0].set_xlabel('Количество чашек кофе', fontsize=11)
    axes[0, 0].set_ylabel('Степень принадлежности', fontsize=11)
    axes[0, 0].set_title('Входное нечеткое множество A*', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # График входных нечетких множеств B*
    axes[0, 1].plot(y_values, B_star_values, 'ks--', linewidth=2, markersize=8, label='B* (не очень поздно)')
    axes[0, 1].set_xlabel('Время отхода ко сну (часы)', fontsize=11)
    axes[0, 1].set_ylabel('Степень принадлежности', fontsize=11)
    axes[0, 1].set_title('Входное нечеткое множество B*', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.1)
    
    # График выходных нечетких множеств
    C1_vals = [mu_C1(z) for z in z_values]
    C2_vals = [mu_C2(z) for z in z_values]
    C3_vals = [mu_C3(z) for z in z_values]
    
    # Создадим отдельную фигуру для выходных множеств
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(z_values, C1_vals, 'ko-', linewidth=2, markersize=8, label='C1 (хорошо)')
    ax2.plot(z_values, C2_vals, 'ks--', linewidth=2, markersize=8, label='C2 (нормально)')
    ax2.plot(z_values, C3_vals, 'k^-', linewidth=2, markersize=8, label='C3 (плохо)')
    ax2.set_xlabel('Вероятность чувствовать себя нормально', fontsize=12)
    ax2.set_ylabel('Степень принадлежности', fontsize=12)
    ax2.set_title('Выходные нечеткие множества', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('task3_output_sets.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График результатов вывода
    axes[1, 0].plot(z_values, mu_res_min, 'ko-', linewidth=2, markersize=8, label='Упрощенный (Max-Min)')
    axes[1, 0].plot(z_values, mu_res_composition, 'ks--', linewidth=2, markersize=8, label='Композиция (Max-Min)')
    axes[1, 0].set_xlabel('Вероятность чувствовать себя нормально', fontsize=11)
    axes[1, 0].set_ylabel('Степень принадлежности', fontsize=11)
    axes[1, 0].set_title('Результат вывода (Max-Min)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.1)
    
    # График сравнения Max-Min и Max-Prod
    axes[1, 1].plot(z_values, mu_res_min, 'ko-', linewidth=2, markersize=8, label='Max-Min')
    axes[1, 1].plot(z_values, mu_res_prod, 'ks--', linewidth=2, markersize=8, label='Max-Prod')
    axes[1, 1].set_xlabel('Вероятность чувствовать себя нормально', fontsize=11)
    axes[1, 1].set_ylabel('Степень принадлежности', fontsize=11)
    axes[1, 1].set_title('Сравнение Max-Min и Max-Prod', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('task3_coffee_wellbeing.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return z_values, mu_res_min, mu_res_composition, mu_res_prod

# Задание 4: Дефаззификация
def task4_defuzzification(z_values, mu_res):
    """
    Дефаззификация нечеткого множества различными методами.
    """
    # Метод центра тяжести (центроид)
    numerator = sum(z * mu for z, mu in zip(z_values, mu_res))
    denominator = sum(mu_res)
    centroid = numerator / denominator if denominator > 0 else 0
    
    # Метод максимума (берем значение с максимальной степенью принадлежности)
    max_idx = np.argmax(mu_res)
    max_value = z_values[max_idx]
    
    # Метод среднего максимума (MOM - Mean of Maximum)
    max_mu = np.max(mu_res)
    max_indices = np.where(mu_res == max_mu)[0]
    mom = np.mean(z_values[max_indices])
    
    # Метод первого максимума (FOM - First of Maximum)
    fom = z_values[max_indices[0]] if len(max_indices) > 0 else max_value
    
    # Метод последнего максимума (LOM - Last of Maximum)
    lom = z_values[max_indices[-1]] if len(max_indices) > 0 else max_value
    
    # Метод взвешенного среднего
    weights = mu_res
    weighted_mean = sum(z * w for z, w in zip(z_values, weights)) / sum(weights) if sum(weights) > 0 else 0
    
    results = {
        'centroid': centroid,
        'max_value': max_value,
        'mom': mom,
        'fom': fom,
        'lom': lom,
        'weighted_mean': weighted_mean
    }
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(z_values, mu_res, 'ko-', linewidth=2, markersize=8, label='Нечеткое множество')
    
    # Отмечаем результаты дефаззификации
    ax.axvline(centroid, color='r', linestyle='--', linewidth=2, label=f'Центроид: {centroid:.3f}')
    ax.axvline(mom, color='b', linestyle='--', linewidth=2, label=f'MOM: {mom:.3f}')
    ax.axvline(weighted_mean, color='g', linestyle='--', linewidth=2, label=f'Взвешенное среднее: {weighted_mean:.3f}')
    
    ax.set_xlabel('Вероятность чувствовать себя нормально', fontsize=12)
    ax.set_ylabel('Степень принадлежности', fontsize=12)
    ax.set_title('Дефаззификация нечеткого множества', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('task4_defuzzification.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

if __name__ == '__main__':
    print("Генерация решений для лабораторной работы 6...")
    
    print("\nЗадание 1: Дискретизация нечетких множеств с PROD...")
    task1_discretization()
    print("✓ График создан: task1_discretization.png")
    
    print("\nЗадание 3: Задача про кофе и самочувствие...")
    z_vals, mu_min, mu_comp, mu_prod = task3_coffee_wellbeing()
    print("✓ График создан: task3_coffee_wellbeing.png")
    
    print("\nЗадание 4: Дефаззификация (Max-Min)...")
    results_min = task4_defuzzification(z_vals, mu_min)
    print(f"Результаты дефаззификации (Max-Min):")
    for method, value in results_min.items():
        print(f"  {method}: {value:.3f}")
    print("✓ График создан: task4_defuzzification.png")
    
    # Сохраняем результаты дефаззификации
    with open('task4_defuzzification_results.txt', 'w', encoding='utf-8') as f:
        f.write("Результаты дефаззификации (Max-Min)\n")
        f.write("=" * 50 + "\n")
        for method, value in results_min.items():
            f.write(f"{method}: {value:.3f}\n")
    
    print("\nЗадание 4: Дефаззификация (Max-Prod)...")
    results_prod = task4_defuzzification(z_vals, mu_prod)
    print(f"Результаты дефаззификации (Max-Prod):")
    for method, value in results_prod.items():
        print(f"  {method}: {value:.3f}")
    
    # Сохраняем результаты дефаззификации Max-Prod
    with open('task4_defuzzification_results_prod.txt', 'w', encoding='utf-8') as f:
        f.write("Результаты дефаззификации (Max-Prod)\n")
        f.write("=" * 50 + "\n")
        for method, value in results_prod.items():
            f.write(f"{method}: {value:.3f}\n")
    
    print("\nВсе задания успешно выполнены!")

