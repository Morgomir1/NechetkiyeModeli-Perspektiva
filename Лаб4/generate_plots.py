import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройка для русского языка
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10

# 1. График функции принадлежности (L-R) числа
def plot_lr_function():
    x = np.linspace(-2, 10, 1000)
    a = 3
    alpha = 1
    beta = 2
    
    mu = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= a:
            u = (a - xi) / alpha
            mu[i] = np.exp(-u**2 / 2)
        else:
            v = (xi - a) / beta
            mu[i] = np.exp(-v**2 / 2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, mu, 'b-', linewidth=2, label='Функция принадлежности')
    plt.axvline(x=a, color='r', linestyle='--', linewidth=1, label=f'Мода a={a}')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('μ(x)', fontsize=12)
    plt.title('График функции принадлежности нечеткого числа (L-R) типа', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.savefig('lr_function.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Графики операций с W-числами
def plot_w_numbers_operations():
    r = np.linspace(0, 1, 100)
    
    # W1 = (1, 3, 7)
    w1_left = 1 + 2*r
    w1_right = 7 - 4*r
    
    # W2 = (4, 8, 9)
    w2_left = 4 + 4*r
    w2_right = 9 - r
    
    # Операции
    sum_left = w1_left + w2_left
    sum_right = w1_right + w2_right
    
    diff_left = w1_left - w2_left
    diff_right = w1_right - w2_right
    
    prod_left = w1_left * w2_left
    prod_right = w1_right * w2_right
    
    div_left = w1_left / w2_left
    div_right = w1_right / w2_right
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Сумма
    axes[0, 0].plot(r, sum_left, 'b-', linewidth=2, label='Левая компонента')
    axes[0, 0].plot(r, sum_right, 'r-', linewidth=2, label='Правая компонента')
    axes[0, 0].set_xlabel('r', fontsize=11)
    axes[0, 0].set_ylabel('Значение', fontsize=11)
    axes[0, 0].set_title('W1 + W2', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Разность
    axes[0, 1].plot(r, diff_left, 'b-', linewidth=2, label='Левая компонента')
    axes[0, 1].plot(r, diff_right, 'r-', linewidth=2, label='Правая компонента')
    axes[0, 1].set_xlabel('r', fontsize=11)
    axes[0, 1].set_ylabel('Значение', fontsize=11)
    axes[0, 1].set_title('W1 - W2', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Произведение
    axes[1, 0].plot(r, prod_left, 'b-', linewidth=2, label='Левая компонента')
    axes[1, 0].plot(r, prod_right, 'r-', linewidth=2, label='Правая компонента')
    axes[1, 0].set_xlabel('r', fontsize=11)
    axes[1, 0].set_ylabel('Значение', fontsize=11)
    axes[1, 0].set_title('W1 · W2', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Частное
    axes[1, 1].plot(r, div_left, 'b-', linewidth=2, label='Левая компонента')
    axes[1, 1].plot(r, div_right, 'r-', linewidth=2, label='Правая компонента')
    axes[1, 1].set_xlabel('r', fontsize=11)
    axes[1, 1].set_ylabel('Значение', fontsize=11)
    axes[1, 1].set_title('W1 : W2', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('w_numbers_operations.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Генерация графиков...")
    plot_lr_function()
    print("✓ График функции принадлежности (L-R) числа создан")
    plot_w_numbers_operations()
    print("✓ Графики операций с W-числами созданы")
    print("Готово!")

