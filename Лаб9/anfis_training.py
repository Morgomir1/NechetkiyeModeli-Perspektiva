# -*- coding: utf-8 -*-
import sys
import io
import os

# Настройка кодировки для Windows консоли (более безопасный способ)
if sys.platform == 'win32':
    # Устанавливаем переменную окружения для кодировки
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Перенастраиваем только если это не subprocess
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass
    if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ANFIS:
    """
    Адаптивная нейро-нечеткая система вывода (ANFIS)
    Упрощенная реализация для системы Такаги-Сугено
    """
    
    def __init__(self, n_inputs=2, n_membership=3, n_rules=None):
        """
        Инициализация ANFIS
        
        Parameters:
        -----------
        n_inputs : int
            Количество входных переменных
        n_membership : int
            Количество функций принадлежности на каждую переменную
        n_rules : int
            Количество правил (по умолчанию n_membership^n_inputs)
        """
        self.n_inputs = n_inputs
        self.n_membership = n_membership
        self.n_rules = n_rules if n_rules else n_membership ** n_inputs
        
        # Параметры функций принадлежности (гауссовы)
        # Для каждой переменной и каждой функции принадлежности: [center, sigma]
        self.membership_params = []
        for i in range(n_inputs):
            var_params = []
            for j in range(n_membership):
                var_params.append([0.0, 1.0])  # [center, sigma]
            self.membership_params.append(var_params)
        
        # Параметры выходных функций (линейные: a0 + a1*x1 + a2*x2 + ...)
        # Для каждого правила: [a0, a1, a2, ...]
        self.output_params = []
        for i in range(self.n_rules):
            self.output_params.append([0.0] * (n_inputs + 1))  # +1 для свободного члена
    
    def gaussian_mf(self, x, center, sigma):
        """Гауссова функция принадлежности"""
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def compute_membership(self, x, var_idx, mf_idx):
        """Вычисление значения функции принадлежности"""
        center, sigma = self.membership_params[var_idx][mf_idx]
        return self.gaussian_mf(x, center, sigma)
    
    def compute_rule_activation(self, inputs, rule_idx):
        """Вычисление степени активации правила"""
        activation = 1.0
        # Преобразуем индекс правила в индексы функций принадлежности
        for var_idx in range(self.n_inputs):
            mf_idx = (rule_idx // (self.n_membership ** var_idx)) % self.n_membership
            mu = self.compute_membership(inputs[var_idx], var_idx, mf_idx)
            activation = min(activation, mu)
        return activation
    
    def compute_rule_output(self, inputs, rule_idx):
        """Вычисление выходного значения правила"""
        params = self.output_params[rule_idx]
        output = params[0]  # свободный член
        for i in range(self.n_inputs):
            output += params[i + 1] * inputs[i]
        return output
    
    def predict(self, X):
        """Предсказание для набора входных данных"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for x in X:
            numerator = 0.0
            denominator = 0.0
            
            for rule_idx in range(self.n_rules):
                w = self.compute_rule_activation(x, rule_idx)
                z = self.compute_rule_output(x, rule_idx)
                
                numerator += w * z
                denominator += w
            
            if denominator == 0:
                predictions.append(0.0)
            else:
                predictions.append(numerator / denominator)
        
        return np.array(predictions)
    
    def initialize_parameters(self, X, y):
        """Инициализация параметров на основе данных"""
        # Инициализация центров функций принадлежности
        for var_idx in range(self.n_inputs):
            var_min = X[:, var_idx].min()
            var_max = X[:, var_idx].max()
            var_range = var_max - var_min
            
            for mf_idx in range(self.n_membership):
                # Равномерное распределение центров
                center = var_min + (var_max - var_min) * (mf_idx + 1) / (self.n_membership + 1)
                sigma = var_range / (2 * self.n_membership)
                self.membership_params[var_idx][mf_idx] = [center, sigma]
        
        # Инициализация параметров выходных функций (линейная регрессия)
        # Упрощенная инициализация: случайные малые значения
        np.random.seed(42)
        for rule_idx in range(self.n_rules):
            self.output_params[rule_idx] = np.random.normal(0, 0.1, self.n_inputs + 1).tolist()
    
    def train_gradient_descent(self, X, y, epochs=100, learning_rate=0.01):
        """
        Обучение ANFIS методом градиентного спуска
        Упрощенная версия
        """
        n_samples = X.shape[0]
        history = {'loss': []}
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Градиенты для параметров
            grad_membership = [[[0.0, 0.0] for _ in range(self.n_membership)] 
                              for _ in range(self.n_inputs)]
            grad_output = [[0.0] * (self.n_inputs + 1) for _ in range(self.n_rules)]
            
            for sample_idx in range(n_samples):
                x = X[sample_idx]
                y_true = y[sample_idx]
                
                # Прямой проход
                numerator = 0.0
                denominator = 0.0
                rule_activations = []
                rule_outputs = []
                
                for rule_idx in range(self.n_rules):
                    w = self.compute_rule_activation(x, rule_idx)
                    z = self.compute_rule_output(x, rule_idx)
                    
                    rule_activations.append(w)
                    rule_outputs.append(z)
                    
                    numerator += w * z
                    denominator += w
                
                if denominator == 0:
                    continue
                
                y_pred = numerator / denominator
                error = y_pred - y_true
                total_loss += error ** 2
                
                # Обратный проход (упрощенный)
                for rule_idx in range(self.n_rules):
                    w = rule_activations[rule_idx]
                    z = rule_outputs[rule_idx]
                    
                    # Градиент по выходным параметрам
                    if denominator > 0:
                        grad_z = w / denominator
                        grad_output[rule_idx][0] += error * grad_z  # по a0
                        for i in range(self.n_inputs):
                            grad_output[rule_idx][i + 1] += error * grad_z * x[i]
            
            # Обновление параметров
            for rule_idx in range(self.n_rules):
                for i in range(self.n_inputs + 1):
                    self.output_params[rule_idx][i] -= learning_rate * grad_output[rule_idx][i] / n_samples
            
            avg_loss = total_loss / n_samples
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Эпоха {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return history
    
    def train_least_squares(self, X, y):
        """
        Обучение выходных параметров методом наименьших квадратов
        (более эффективный метод для линейных выходов)
        """
        n_samples = X.shape[0]
        
        # Формирование матрицы признаков
        # Каждая строка: [w1, w1*x1, w1*x2, w2, w2*x1, w2*x2, ...]
        features = []
        
        for sample_idx in range(n_samples):
            x = X[sample_idx]
            row = []
            
            for rule_idx in range(self.n_rules):
                w = self.compute_rule_activation(x, rule_idx)
                denominator = sum([self.compute_rule_activation(x, r) for r in range(self.n_rules)])
                
                if denominator > 0:
                    w_normalized = w / denominator
                else:
                    w_normalized = 0.0
                
                # Добавляем признаки для этого правила: [w, w*x1, w*x2, ...]
                row.append(w_normalized)
                for i in range(self.n_inputs):
                    row.append(w_normalized * x[i])
            
            features.append(row)
        
        features = np.array(features)
        
        # Решение методом наименьших квадратов
        try:
            params = np.linalg.lstsq(features, y, rcond=None)[0]
            
            # Распределение параметров по правилам
            param_idx = 0
            for rule_idx in range(self.n_rules):
                self.output_params[rule_idx][0] = params[param_idx]  # a0
                param_idx += 1
                for i in range(self.n_inputs):
                    self.output_params[rule_idx][i + 1] = params[param_idx]  # ai
                    param_idx += 1
        except:
            print("Предупреждение: не удалось решить систему методом наименьших квадратов")


def load_credit_data():
    """Загрузка данных Credit"""
    import pickle
    try:
        with open('sugeno_credit_results.pkl', 'rb') as f:
            data = pickle.load(f)
            return data['data']
    except:
        # Если файл не найден, генерируем данные
        from sugeno_credit_system import generate_credit_data
        credit_data, _, _ = generate_credit_data()
        return credit_data


if __name__ == '__main__':
    # Загрузка данных
    print("Загрузка данных Credit...")
    credit_data = load_credit_data()
    
    # Подготовка данных для обучения
    # Используем возраст и доход как входные переменные
    X = credit_data[['возраст', 'доход']].values
    y = credit_data['оценка_кредита'].values
    
    # Нормализация данных
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )
    
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    
    # Создание и инициализация ANFIS
    print("\nСоздание ANFIS...")
    anfis = ANFIS(n_inputs=2, n_membership=3, n_rules=9)
    anfis.initialize_parameters(X_train, y_train)
    
    # Обучение ANFIS
    # Сначала обучаем градиентным спуском для видимой динамики обучения
    print("\nОбучение ANFIS градиентным спуском...")
    history = anfis.train_gradient_descent(X_train, y_train, epochs=100, learning_rate=0.01)
    
    # Вычисляем loss после градиентного спуска
    y_pred_after_gd = anfis.predict(X_train)
    loss_after_gd = np.mean((y_pred_after_gd - y_train) ** 2)
    print(f"Loss после градиентного спуска: {loss_after_gd:.6f}")
    
    # Затем дообучаем методом наименьших квадратов для улучшения результата
    print("\nДообучение ANFIS методом наименьших квадратов...")
    anfis.train_least_squares(X_train, y_train)
    
    # Вычисляем loss после метода наименьших квадратов
    y_pred_after_ls = anfis.predict(X_train)
    loss_after_ls = np.mean((y_pred_after_ls - y_train) ** 2)
    history['loss'].append(loss_after_ls)  # Добавляем точку после метода наименьших квадратов
    print(f"Loss после метода наименьших квадратов: {loss_after_ls:.6f}")
    
    # Финальное дообучение градиентным спуском с меньшим learning rate
    print("\nФинальное дообучение градиентным спуском...")
    final_history = anfis.train_gradient_descent(X_train, y_train, epochs=20, learning_rate=0.001)
    
    # Объединяем истории обучения
    history['loss'].extend(final_history['loss'])
    
    # Убеждаемся, что история в правильном формате (список, а не numpy массив)
    if history and 'loss' in history:
        if hasattr(history['loss'], 'tolist'):
            history['loss'] = history['loss'].tolist()
        elif not isinstance(history['loss'], list):
            history['loss'] = list(history['loss'])
    
    # Предсказания
    y_train_pred = anfis.predict(X_train)
    y_test_pred = anfis.predict(X_test)
    
    # Обратное масштабирование
    y_train_pred_orig = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
    y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
    y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # Метрики
    train_mse = mean_squared_error(y_train_orig, y_train_pred_orig)
    test_mse = mean_squared_error(y_test_orig, y_test_pred_orig)
    train_r2 = r2_score(y_train_orig, y_train_pred_orig)
    test_r2 = r2_score(y_test_orig, y_test_pred_orig)
    
    print("\n" + "=" * 60)
    print("Результаты обучения ANFIS:")
    print("=" * 60)
    print(f"Обучающая выборка:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  R^2: {train_r2:.6f}")
    print(f"\nТестовая выборка:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  R^2: {test_r2:.6f}")
    
    # Сохранение результатов
    import pickle
    
    # Убеждаемся, что все данные в правильном формате
    print("\nПроверка данных перед сохранением:")
    print(f"  history type: {type(history)}, keys: {list(history.keys()) if isinstance(history, dict) else 'N/A'}")
    print(f"  y_train_orig shape: {y_train_orig.shape if hasattr(y_train_orig, 'shape') else 'N/A'}")
    print(f"  y_test_orig shape: {y_test_orig.shape if hasattr(y_test_orig, 'shape') else 'N/A'}")
    print(f"  y_train_pred_orig shape: {y_train_pred_orig.shape if hasattr(y_train_pred_orig, 'shape') else 'N/A'}")
    print(f"  y_test_pred_orig shape: {y_test_pred_orig.shape if hasattr(y_test_pred_orig, 'shape') else 'N/A'}")
    
    results = {
        'anfis': anfis,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'history': history,
        'X_train': X_train,
        'y_train': y_train,
        'y_train_orig': y_train_orig,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_orig': y_test_orig,
        'y_train_pred': y_train_pred_orig,
        'y_test_pred': y_test_pred_orig
    }
    
    # Проверяем наличие всех ключей
    required_keys = ['history', 'y_train_orig', 'y_test_orig', 'y_train_pred', 'y_test_pred']
    missing_keys = [key for key in required_keys if key not in results or results[key] is None]
    if missing_keys:
        print(f"  [WARN] ВНИМАНИЕ: Отсутствующие ключи: {missing_keys}")
    else:
        print(f"  ✓ Все необходимые ключи присутствуют")
    
    with open('anfis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nРезультаты сохранены в anfis_results.pkl")
    print(f"Сохранено ключей: {len(results)}")
