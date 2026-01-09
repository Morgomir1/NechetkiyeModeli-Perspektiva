import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class SugenoFuzzySystem:
    """Класс для реализации нечеткой системы Суджено"""
    
    def __init__(self):
        self.rules = []
        self.input_variables = {}
        self.output_functions = {}
        
    def add_input_variable(self, name, universe, membership_functions):
        """
        Добавление входной переменной
        
        Parameters:
        -----------
        name : str
            Имя переменной
        universe : array-like
            Универсум (диапазон значений)
        membership_functions : dict
            Словарь функций принадлежности {терм: функция}
        """
        self.input_variables[name] = {
            'universe': np.array(universe),
            'membership_functions': membership_functions
        }
    
    def add_rule(self, conditions, output_function):
        """
        Добавление правила
        
        Parameters:
        -----------
        conditions : dict
            Словарь условий {переменная: терм}
        output_function : str или callable
            Выходная функция (константа или функция входных переменных)
        """
        self.rules.append({
            'conditions': conditions,
            'output': output_function
        })
    
    def membership_value(self, variable_name, term, x):
        """Вычисление значения функции принадлежности"""
        if variable_name not in self.input_variables:
            return 0.0
        
        universe = self.input_variables[variable_name]['universe']
        mf = self.input_variables[variable_name]['membership_functions'][term]
        
        # Интерполяция функции принадлежности
        if callable(mf):
            return mf(x)
        else:
            # Если mf - это массив значений, используем numpy.interp
            if x < universe.min() or x > universe.max():
                return 0.0
            return float(np.interp(x, universe, mf))
    
    def evaluate_rule(self, rule, inputs):
        """Вычисление степени активации правила"""
        activation = 1.0
        for var_name, term in rule['conditions'].items():
            if var_name in inputs:
                mu = self.membership_value(var_name, term, inputs[var_name])
                activation = min(activation, mu)
            else:
                return 0.0
        return activation
    
    def compute_output(self, rule, inputs):
        """Вычисление выходного значения правила"""
        output_func = rule['output']
        if callable(output_func):
            return output_func(inputs)
        elif isinstance(output_func, (int, float)):
            return output_func
        else:
            # Если это строка с формулой, парсим её
            # Для простоты, будем использовать eval (в реальности нужен парсер)
            try:
                # Заменяем имена переменных на их значения
                local_vars = inputs.copy()
                return eval(output_func, {"__builtins__": {}}, local_vars)
            except:
                return 0.0
    
    def infer(self, inputs):
        """
        Вывод Суджено
        
        Parameters:
        -----------
        inputs : dict
            Словарь входных значений {переменная: значение}
        
        Returns:
        --------
        output : float
            Выходное значение системы
        """
        numerator = 0.0
        denominator = 0.0
        
        for rule in self.rules:
            w = self.evaluate_rule(rule, inputs)  # Степень активации
            z = self.compute_output(rule, inputs)  # Выходное значение правила
            
            numerator += w * z
            denominator += w
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


def triangular_mf(a, b, c):
    """Треугольная функция принадлежности"""
    def mf(x):
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    return mf


def trapezoidal_mf(a, b, c, d):
    """Трапециевидная функция принадлежности"""
    def mf(x):
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c)
    return mf


def gaussian_mf(center, sigma):
    """Гауссова функция принадлежности"""
    def mf(x):
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return mf


# Создание системы Суджено для управления температурой
# Входные переменные: температура (T) и скорость изменения температуры (dT)
# Выходная переменная: мощность нагревателя (P)

# Определение универсумов
T_universe = np.linspace(0, 50, 1000)  # Температура от 0 до 50 градусов
dT_universe = np.linspace(-5, 5, 1000)  # Скорость изменения от -5 до +5 град/мин

# Создание системы
system = SugenoFuzzySystem()

# Определение функций принадлежности для температуры
T_low = triangular_mf(0, 0, 20)
T_medium = triangular_mf(10, 25, 40)
T_high = triangular_mf(30, 50, 50)

# Определение функций принадлежности для скорости изменения температуры
dT_negative = triangular_mf(-5, -5, 0)
dT_zero = triangular_mf(-2.5, 0, 2.5)
dT_positive = triangular_mf(0, 5, 5)

# Добавление входных переменных
system.add_input_variable('T', T_universe, {
    'low': np.array([T_low(x) for x in T_universe]),
    'medium': np.array([T_medium(x) for x in T_universe]),
    'high': np.array([T_high(x) for x in T_universe])
})

system.add_input_variable('dT', dT_universe, {
    'negative': np.array([dT_negative(x) for x in dT_universe]),
    'zero': np.array([dT_zero(x) for x in dT_universe]),
    'positive': np.array([dT_positive(x) for x in dT_universe])
})

# Определение правил Суджено
# В системе Суджено выходные значения - это функции входных переменных или константы

# Правило 1: Если T низкая И dT отрицательная, то P = 100 (максимальная мощность)
system.add_rule({'T': 'low', 'dT': 'negative'}, 100)

# Правило 2: Если T низкая И dT нулевая, то P = 80
system.add_rule({'T': 'low', 'dT': 'zero'}, 80)

# Правило 3: Если T низкая И dT положительная, то P = 60
system.add_rule({'T': 'low', 'dT': 'positive'}, 60)

# Правило 4: Если T средняя И dT отрицательная, то P = 70
system.add_rule({'T': 'medium', 'dT': 'negative'}, 70)

# Правило 5: Если T средняя И dT нулевая, то P = 50
system.add_rule({'T': 'medium', 'dT': 'zero'}, 50)

# Правило 6: Если T средняя И dT положительная, то P = 30
system.add_rule({'T': 'medium', 'dT': 'positive'}, 30)

# Правило 7: Если T высокая И dT отрицательная, то P = 40
system.add_rule({'T': 'high', 'dT': 'negative'}, 40)

# Правило 8: Если T высокая И dT нулевая, то P = 20
system.add_rule({'T': 'high', 'dT': 'zero'}, 20)

# Правило 9: Если T высокая И dT положительная, то P = 0 (минимальная мощность)
system.add_rule({'T': 'high', 'dT': 'positive'}, 0)

# Тестирование системы
print("Тестирование системы Суджено")
print("=" * 60)

test_cases = [
    {'T': 5, 'dT': -2},
    {'T': 15, 'dT': 0},
    {'T': 25, 'dT': 1},
    {'T': 35, 'dT': -1},
    {'T': 45, 'dT': 3},
]

results = []
for test in test_cases:
    output = system.infer(test)
    results.append({
        'T': test['T'],
        'dT': test['dT'],
        'P': output
    })
    print(f"T = {test['T']:.1f}°C, dT = {test['dT']:.1f} град/мин -> P = {output:.2f}")

# Сохранение результатов
import pickle
with open('sugeno_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'test_cases': test_cases,
        'system': system
    }, f)

print("\nРезультаты сохранены в sugeno_results.pkl")
