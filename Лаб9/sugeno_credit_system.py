import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class SugenoFuzzySystem:
    """Класс для реализации нечеткой системы Такаги-Сугено"""
    
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
            try:
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


# Генерация данных Credit (аналогично работе по деревьям решений)
def generate_credit_data():
    """Генерация синтетических данных о кредитах"""
    np.random.seed(42)
    n_samples = 1000
    
    # Генерация признаков
    age = np.random.randint(18, 70, n_samples)
    income = np.random.randint(20000, 150000, n_samples)
    credit_history = np.random.choice(['хорошая', 'средняя', 'плохая'], n_samples)
    loan_purpose = np.random.choice(['потребительский', 'автомобиль', 'недвижимость', 'бизнес'], n_samples)
    loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)
    
    # Создание целевой переменной (вероятность возврата кредита: 0-1)
    def generate_credit_score(age, income, credit_history, loan_purpose, loan_term):
        score = 0.5
        
        # Возраст: оптимальный 30-50 лет
        if 30 <= age <= 50:
            score += 0.15
        elif age < 25 or age > 60:
            score -= 0.1
        
        # Доход: чем выше, тем лучше
        if income > 80000:
            score += 0.2
        elif income < 40000:
            score -= 0.15
        
        # Кредитная история
        if credit_history == 'хорошая':
            score += 0.25
        elif credit_history == 'средняя':
            score += 0.05
        else:
            score -= 0.2
        
        # Цель кредита
        if loan_purpose == 'недвижимость':
            score += 0.1
        elif loan_purpose == 'бизнес':
            score -= 0.1
        
        # Срок кредита: короткие лучше
        if loan_term <= 24:
            score += 0.1
        elif loan_term >= 48:
            score -= 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    credit_score = [generate_credit_score(age[i], income[i], credit_history[i], 
                                          loan_purpose[i], loan_term[i]) 
                    for i in range(n_samples)]
    
    # Кодирование категориальных переменных
    from sklearn.preprocessing import LabelEncoder
    le_history = LabelEncoder()
    le_purpose = LabelEncoder()
    
    credit_history_encoded = le_history.fit_transform(credit_history)
    loan_purpose_encoded = le_purpose.fit_transform(loan_purpose)
    
    data = pd.DataFrame({
        'возраст': age,
        'доход': income,
        'кредитная_история': credit_history_encoded,
        'цель_кредита': loan_purpose_encoded,
        'срок_кредита': loan_term,
        'оценка_кредита': credit_score
    })
    
    return data, le_history, le_purpose


# Создание системы Такаги-Сугено для Credit
def create_credit_sugeno_system():
    """Создание нечеткой системы Такаги-Сугено для оценки кредита"""
    
    # Определение универсумов
    age_universe = np.linspace(18, 70, 1000)
    income_universe = np.linspace(20000, 150000, 1000)
    
    # Создание системы
    system = SugenoFuzzySystem()
    
    # Определение функций принадлежности для возраста
    age_young = triangular_mf(18, 18, 35)
    age_middle = triangular_mf(25, 45, 65)
    age_old = triangular_mf(55, 70, 70)
    
    # Определение функций принадлежности для дохода
    income_low = triangular_mf(20000, 20000, 60000)
    income_medium = triangular_mf(40000, 80000, 120000)
    income_high = triangular_mf(100000, 150000, 150000)
    
    # Добавление входных переменных
    system.add_input_variable('возраст', age_universe, {
        'молодой': np.array([age_young(x) for x in age_universe]),
        'средний': np.array([age_middle(x) for x in age_universe]),
        'пожилой': np.array([age_old(x) for x in age_universe])
    })
    
    system.add_input_variable('доход', income_universe, {
        'низкий': np.array([income_low(x) for x in income_universe]),
        'средний': np.array([income_medium(x) for x in income_universe]),
        'высокий': np.array([income_high(x) for x in income_universe])
    })
    
    # Определение функций вывода для правил
    def rule1_output(inputs):
        """Правило 1: молодой + низкий доход"""
        return 0.2 + 0.000001 * inputs['доход']
    
    def rule2_output(inputs):
        """Правило 2: молодой + средний доход"""
        return 0.4 + 0.000002 * inputs['доход']
    
    def rule3_output(inputs):
        """Правило 3: молодой + высокий доход"""
        return 0.6 + 0.000001 * inputs['доход']
    
    def rule4_output(inputs):
        """Правило 4: средний возраст + низкий доход"""
        return 0.3 + 0.0000015 * inputs['доход']
    
    def rule5_output(inputs):
        """Правило 5: средний возраст + средний доход"""
        return 0.5 + 0.000002 * inputs['доход']
    
    def rule6_output(inputs):
        """Правило 6: средний возраст + высокий доход"""
        return 0.7 + 0.0000015 * inputs['доход']
    
    def rule7_output(inputs):
        """Правило 7: пожилой + низкий доход"""
        return 0.25 + 0.000001 * inputs['доход']
    
    def rule8_output(inputs):
        """Правило 8: пожилой + средний доход"""
        return 0.45 + 0.0000015 * inputs['доход']
    
    def rule9_output(inputs):
        """Правило 9: пожилой + высокий доход"""
        return 0.65 + 0.000001 * inputs['доход']
    
    # Определение правил Такаги-Сугено
    # Выходные значения - линейные функции входных переменных
    
    # Правило 1: Если возраст молодой И доход низкий, то оценка = 0.2 + 0.000001*доход
    system.add_rule({'возраст': 'молодой', 'доход': 'низкий'}, rule1_output)
    
    # Правило 2: Если возраст молодой И доход средний, то оценка = 0.4 + 0.000002*доход
    system.add_rule({'возраст': 'молодой', 'доход': 'средний'}, rule2_output)
    
    # Правило 3: Если возраст молодой И доход высокий, то оценка = 0.6 + 0.000001*доход
    system.add_rule({'возраст': 'молодой', 'доход': 'высокий'}, rule3_output)
    
    # Правило 4: Если возраст средний И доход низкий, то оценка = 0.3 + 0.0000015*доход
    system.add_rule({'возраст': 'средний', 'доход': 'низкий'}, rule4_output)
    
    # Правило 5: Если возраст средний И доход средний, то оценка = 0.5 + 0.000002*доход
    system.add_rule({'возраст': 'средний', 'доход': 'средний'}, rule5_output)
    
    # Правило 6: Если возраст средний И доход высокий, то оценка = 0.7 + 0.0000015*доход
    system.add_rule({'возраст': 'средний', 'доход': 'высокий'}, rule6_output)
    
    # Правило 7: Если возраст пожилой И доход низкий, то оценка = 0.25 + 0.000001*доход
    system.add_rule({'возраст': 'пожилой', 'доход': 'низкий'}, rule7_output)
    
    # Правило 8: Если возраст пожилой И доход средний, то оценка = 0.45 + 0.0000015*доход
    system.add_rule({'возраст': 'пожилой', 'доход': 'средний'}, rule8_output)
    
    # Правило 9: Если возраст пожилой И доход высокий, то оценка = 0.65 + 0.000001*доход
    system.add_rule({'возраст': 'пожилой', 'доход': 'высокий'}, rule9_output)
    
    return system


if __name__ == '__main__':
    # Генерация данных
    print("Генерация данных Credit...")
    credit_data, le_history, le_purpose = generate_credit_data()
    
    # Создание системы
    print("Создание системы Такаги-Сугено...")
    sugeno_system = create_credit_sugeno_system()
    
    # Тестирование системы
    print("\nТестирование системы Такаги-Сугено")
    print("=" * 60)
    
    test_cases = [
        {'возраст': 25, 'доход': 30000},
        {'возраст': 35, 'доход': 70000},
        {'возраст': 45, 'доход': 100000},
        {'возраст': 55, 'доход': 50000},
        {'возраст': 30, 'доход': 120000},
    ]
    
    results = []
    for test in test_cases:
        output = sugeno_system.infer(test)
        results.append({
            'возраст': test['возраст'],
            'доход': test['доход'],
            'оценка': output
        })
        print(f"Возраст = {test['возраст']:.0f} лет, Доход = {test['доход']:.0f} руб -> Оценка = {output:.4f}")
    
    # Сохранение результатов
    import pickle
    with open('sugeno_credit_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'test_cases': test_cases,
            'data': credit_data
            # 'system': sugeno_system,  # Не сохраняем систему, т.к. содержит функции, которые нельзя сериализовать
        }, f)
    
    print("\nРезультаты сохранены в sugeno_credit_results.pkl")
