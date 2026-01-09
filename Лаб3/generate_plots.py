import matplotlib.pyplot as plt
import numpy as np

# Настройка для цветного вывода
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Задание 1: Множество B+
def plot_task1():
    fig, ax = plt.subplots(figsize=(8, 6))
    x = [1, 0, -1, -2, -3, -4, -5]
    y = [0, 0.8, 0.6, 0.4, 0.2, 0.1]
    # Исправление: B+ = {0/1; -1/0.8; -2/0.6; -3/0.4; -4/0.2; -5/0.1}
    x_vals = [0, -1, -2, -3, -4, -5]
    mu_vals = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
    
    ax.stem(x_vals, mu_vals, basefmt=' ', linefmt='b-', markerfmt='bo')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('μ(x)', fontsize=12)
    ax.set_title('Задание 1: Множество B+', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('task1.png', dpi=150, bbox_inches='tight')
    plt.close()

# Задание 2: Графики функций μA и μB-
def plot_task2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # μA
    x_a = np.linspace(0, 4, 1000)
    mu_a = np.zeros_like(x_a)
    mu_a[(x_a >= 1) & (x_a <= 2)] = x_a[(x_a >= 1) & (x_a <= 2)] - 1
    mu_a[(x_a > 2) & (x_a <= 3)] = 3 - x_a[(x_a > 2) & (x_a <= 3)]
    
    ax1.plot(x_a, mu_a, 'b-', linewidth=2, color='blue')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('μ_A(x)', fontsize=12)
    ax1.set_title('Функция μ_A', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # μB-
    x_b = np.linspace(-4, 0, 1000)
    mu_b = np.zeros_like(x_b)
    mu_b[(x_b >= -2) & (x_b <= -1)] = -x_b[(x_b >= -2) & (x_b <= -1)] - 1
    mu_b[(x_b > -3) & (x_b <= -2)] = 3 + x_b[(x_b > -3) & (x_b <= -2)]
    
    ax2.plot(x_b, mu_b, 'r-', linewidth=2, color='red')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('μ_B-(x)', fontsize=12)
    ax2.set_title('Функция μ_B-', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('task2.png', dpi=150, bbox_inches='tight')
    plt.close()

# Задание 3: Примерно 2, примерно 3 и их сложение
def plot_task3():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Примерно 2
    x_approx2 = [0, 1, 2, 3, 4]
    mu_approx2 = [0.2, 0.6, 1, 0.6, 0.2]
    
    # Примерно 3
    x_approx3 = [1, 2, 3, 4, 5]
    mu_approx3 = [0.1, 0.7, 1, 0.7, 0.1]
    
    # Результат сложения
    x_sum = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    mu_sum = [0.1, 0.2, 0.6, 0.7, 1.0, 0.7, 0.6, 0.2, 0.1]
    
    ax.plot(x_approx2, mu_approx2, linewidth=2, marker='o', label='Примерно 2', linestyle='-', color='blue')
    ax.plot(x_approx3, mu_approx3, linewidth=2, marker='s', label='Примерно 3', linestyle='--', color='green')
    ax.plot(x_sum, mu_sum, linewidth=2, marker='^', label='Сумма', linestyle='-.', color='red')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('μ(x)', fontsize=12)
    ax.set_title('Задание 3: Примерно 2, примерно 3 и их сложение', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('task3.png', dpi=150, bbox_inches='tight')
    plt.close()

# Задание 4: Нечеткая 2 и Нечеткая -2
def plot_task4():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Нечеткая 2
    x_fuzzy2 = [0, 1, 2, 3, 4]
    mu_fuzzy2 = [0.2, 0.6, 1, 0.6, 0.2]
    
    # Нечеткая -2
    x_fuzzy_minus2 = [-4, -3, -2, -1, 0]
    mu_fuzzy_minus2 = [0.2, 0.6, 1, 0.6, 0.2]
    
    # Результат сложения
    x_sum = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    mu_sum = [0.2, 0.2, 0.6, 0.6, 1.0, 0.6, 0.6, 0.2, 0.2]
    
    ax.plot(x_fuzzy2, mu_fuzzy2, linewidth=2, marker='o', label='Нечеткая 2', linestyle='-', color='blue')
    ax.plot(x_fuzzy_minus2, mu_fuzzy_minus2, linewidth=2, marker='s', label='Нечеткая -2', linestyle='--', color='orange')
    ax.plot(x_sum, mu_sum, linewidth=2, marker='^', label='Сумма', linestyle='-.', color='red')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('μ(x)', fontsize=12)
    ax.set_title('Задание 4: Нечеткая 2, Нечеткая -2 и их сумма', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('task4.png', dpi=150, bbox_inches='tight')
    plt.close()

# Задание 7: α-уровни для операций
def plot_task7():
    # Данные из таблицы
    alpha_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Сложение
    add_left = [10.7, 11.4, 12.8, 14.2, 15.6, 17]
    add_right = [23.3, 22.6, 21.2, 19.8, 18.4, 17]
    
    # Вычитание
    sub_left = [-10.2, -9.4, -7.8, -6.2, -4.6, -3]
    sub_right = [2.4, 1.8, 0.6, -0.6, -1.8, -3]
    
    # Умножение
    mult_left = [27.52, 31.28, 39.52, 48.72, 58.88, 70]
    mult_right = [127.6, 120.4, 106.6, 93.6, 81.4, 70]
    
    # Деление (приблизительные значения)
    div_left = [43/145, 23/70, 2/5, 29/60, 32/55, 7/10]
    div_right = [11/8, 43/32, 41/38, 13/14, 37/46, 7/10]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Сложение
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, alpha in enumerate(alpha_values):
        ax1.plot([add_left[i], add_right[i]], [alpha, alpha], linewidth=2, color=colors[i])
        ax1.plot(add_left[i], alpha, 'o', markersize=6, color=colors[i])
        ax1.plot(add_right[i], alpha, 'o', markersize=6, color=colors[i])
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('α', fontsize=12)
    ax1.set_title('Сложение', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Вычитание
    for i, alpha in enumerate(alpha_values):
        ax2.plot([sub_left[i], sub_right[i]], [alpha, alpha], linewidth=2, color=colors[i])
        ax2.plot(sub_left[i], alpha, 'o', markersize=6, color=colors[i])
        ax2.plot(sub_right[i], alpha, 'o', markersize=6, color=colors[i])
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('α', fontsize=12)
    ax2.set_title('Вычитание', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Умножение
    for i, alpha in enumerate(alpha_values):
        ax3.plot([mult_left[i], mult_right[i]], [alpha, alpha], linewidth=2, color=colors[i])
        ax3.plot(mult_left[i], alpha, 'o', markersize=6, color=colors[i])
        ax3.plot(mult_right[i], alpha, 'o', markersize=6, color=colors[i])
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('α', fontsize=12)
    ax3.set_title('Умножение', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    # Деление
    for i, alpha in enumerate(alpha_values):
        ax4.plot([div_left[i], div_right[i]], [alpha, alpha], linewidth=2, color=colors[i])
        ax4.plot(div_left[i], alpha, 'o', markersize=6, color=colors[i])
        ax4.plot(div_right[i], alpha, 'o', markersize=6, color=colors[i])
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('α', fontsize=12)
    ax4.set_title('Деление', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('task7.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Генерация графиков...")
    plot_task1()
    print("Задание 1 готово")
    plot_task2()
    print("Задание 2 готово")
    plot_task3()
    print("Задание 3 готово")
    plot_task4()
    print("Задание 4 готово")
    plot_task7()
    print("Задание 7 готово")
    print("Все графики сгенерированы!")

