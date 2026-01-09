import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sugeno_fuzzy_system import SugenoFuzzySystem, triangular_mf
import pickle
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Загрузка результатов (будет выполнена в функциях)
_data_cache = None
_system_cache = None

def load_data():
    """Загрузка данных из pickle файла"""
    global _data_cache, _system_cache
    if _data_cache is None:
        with open('sugeno_results.pkl', 'rb') as f:
            _data_cache = pickle.load(f)
            _system_cache = _data_cache['system']
    return _data_cache, _system_cache

# График 1: Функции принадлежности для температуры
def plot_temperature_membership():
    data, system = load_data()
    T_universe = np.linspace(0, 50, 1000)
    
    T_low = triangular_mf(0, 0, 20)
    T_medium = triangular_mf(10, 25, 40)
    T_high = triangular_mf(30, 50, 50)
    
    mu_low = [T_low(x) for x in T_universe]
    mu_medium = [T_medium(x) for x in T_universe]
    mu_high = [T_high(x) for x in T_universe]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(T_universe, mu_low, 'b-', linewidth=2, label='Низкая')
    ax.plot(T_universe, mu_medium, 'g-', linewidth=2, label='Средняя')
    ax.plot(T_universe, mu_high, 'r-', linewidth=2, label='Высокая')
    
    ax.set_xlabel('Температура, °C', fontsize=12)
    ax.set_ylabel('Степень принадлежности μ(T)', fontsize=12)
    ax.set_title('Функции принадлежности для переменной "Температура"', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('temperature_membership.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График функций принадлежности для температуры сохранен")

# График 2: Функции принадлежности для скорости изменения температуры
def plot_dT_membership():
    dT_universe = np.linspace(-5, 5, 1000)
    
    dT_negative = triangular_mf(-5, -5, 0)
    dT_zero = triangular_mf(-2.5, 0, 2.5)
    dT_positive = triangular_mf(0, 5, 5)
    
    mu_negative = [dT_negative(x) for x in dT_universe]
    mu_zero = [dT_zero(x) for x in dT_universe]
    mu_positive = [dT_positive(x) for x in dT_universe]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dT_universe, mu_negative, 'b-', linewidth=2, label='Отрицательная')
    ax.plot(dT_universe, mu_zero, 'g-', linewidth=2, label='Нулевая')
    ax.plot(dT_universe, mu_positive, 'r-', linewidth=2, label='Положительная')
    
    ax.set_xlabel('Скорость изменения температуры, град/мин', fontsize=12)
    ax.set_ylabel('Степень принадлежности μ(dT)', fontsize=12)
    ax.set_title('Функции принадлежности для переменной "Скорость изменения температуры"', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('dT_membership.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График функций принадлежности для скорости изменения температуры сохранен")

# График 3: Поверхность вывода системы Суджено
def plot_sugeno_surface():
    data, system = load_data()
    T_range = np.linspace(0, 50, 30)
    dT_range = np.linspace(-5, 5, 30)
    
    T_grid, dT_grid = np.meshgrid(T_range, dT_range)
    P_grid = np.zeros_like(T_grid)
    
    for i in range(len(T_range)):
        for j in range(len(dT_range)):
            inputs = {'T': T_range[i], 'dT': dT_range[j]}
            P_grid[j, i] = system.infer(inputs)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(T_grid, dT_grid, P_grid, cmap='viridis', 
                           alpha=0.8, linewidth=0, antialiased=True)
    
    ax.set_xlabel('Температура, °C', fontsize=12)
    ax.set_ylabel('Скорость изменения, град/мин', fontsize=12)
    ax.set_zlabel('Мощность нагревателя, %', fontsize=12)
    ax.set_title('Поверхность вывода системы Суджено', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    plt.tight_layout()
    plt.savefig('sugeno_surface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График поверхности вывода системы Суджено сохранен")

# График 4: Результаты тестирования
def plot_test_results():
    data, system = load_data()
    results = data['results']
    
    T_values = [r['T'] for r in results]
    dT_values = [r['dT'] for r in results]
    P_values = [r['P'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График зависимости мощности от температуры
    ax1.plot(T_values, P_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Температура, °C', fontsize=12)
    ax1.set_ylabel('Мощность нагревателя, %', fontsize=12)
    ax1.set_title('Зависимость мощности от температуры', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # График зависимости мощности от скорости изменения
    ax2.plot(dT_values, P_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Скорость изменения температуры, град/мин', fontsize=12)
    ax2.set_ylabel('Мощность нагревателя, %', fontsize=12)
    ax2.set_title('Зависимость мощности от скорости изменения', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График результатов тестирования сохранен")

def generate_all_plots():
    """Функция для генерации всех графиков"""
    print("Генерация графиков...")
    plot_temperature_membership()
    plot_dT_membership()
    plot_sugeno_surface()
    plot_test_results()
    print("Все графики успешно сгенерированы!")

if __name__ == '__main__':
    generate_all_plots()
