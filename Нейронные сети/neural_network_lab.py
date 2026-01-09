import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Установка seed для воспроизводимости
np.random.seed(42)

# Настройка matplotlib для черно-белого вывода
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def generate_concrete_data():
    """Генерация данных о прочности бетона (аналог датасета UCI Concrete)"""
    # Создаем синтетические данные, похожие на реальный датасет
    n_samples = 1030
    np.random.seed(42)
    
    # Генерируем входные признаки (8 признаков)
    cement = np.random.uniform(100, 540, n_samples)
    blast_furnace_slag = np.random.uniform(0, 360, n_samples)
    fly_ash = np.random.uniform(0, 200, n_samples)
    water = np.random.uniform(120, 250, n_samples)
    superplasticizer = np.random.uniform(0, 30, n_samples)
    coarse_aggregate = np.random.uniform(800, 1150, n_samples)
    fine_aggregate = np.random.uniform(600, 1000, n_samples)
    age = np.random.choice([1, 3, 7, 14, 28, 56, 90, 180, 365], n_samples)
    
    # Генерируем целевую переменную (прочность) на основе некоторых зависимостей
    strength = (cement * 0.1 + blast_furnace_slag * 0.05 + 
                (28 / age) * 20 - water * 0.15 + 
                superplasticizer * 0.5 + np.random.normal(0, 10, n_samples))
    strength = np.maximum(strength, 2)  # Минимальная прочность
    strength = np.minimum(strength, 82)  # Максимальная прочность
    
    data = pd.DataFrame({
        'cement': cement,
        'blast_furnace_slag': blast_furnace_slag,
        'fly_ash': fly_ash,
        'water': water,
        'superplasticizer': superplasticizer,
        'coarse_aggregate': coarse_aggregate,
        'fine_aggregate': fine_aggregate,
        'age': age,
        'strength': strength
    })
    
    return data

def normalize_data(data):
    """Нормализация данных к диапазону [0, 1]"""
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    return normalized, scaler

def denormalize_data(normalized_data, scaler):
    """Денормализация данных"""
    return scaler.inverse_transform(normalized_data)

# Загрузка данных
print("Загрузка данных...")
concrete_data = generate_concrete_data()

# Разделение на признаки и целевую переменную
X = concrete_data.drop('strength', axis=1).values
y = concrete_data['strength'].values.reshape(-1, 1)

# Нормализация данных
X_norm, scaler_X = normalize_data(X)
y_norm, scaler_y = normalize_data(y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_norm, test_size=0.25, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

# Сохранение данных для отчета
results = {}

# Модель 1: Один скрытый узел
print("\nОбучение модели 1: 1 скрытый узел...")
model1 = MLPRegressor(
    hidden_layer_sizes=(1,),
    activation='logistic',  # сигмоида
    solver='adam',
    max_iter=1000,
    random_state=42,
    tol=1e-6
)
model1.fit(X_train, y_train.ravel())

# Предсказания
y_pred1 = model1.predict(X_test).reshape(-1, 1)
correlation1 = np.corrcoef(y_test.flatten(), y_pred1.flatten())[0, 1]

results['model1'] = {
    'weights': model1.coefs_,
    'intercepts': model1.intercepts_,
    'correlation': correlation1
}

print(f"Корреляция модели 1: {correlation1:.6f}")
print(f"Количество итераций: {model1.n_iter_}")
print(f"Потеря: {model1.loss_:.6f}")

# Модель 2: Три скрытых узла
print("\nОбучение модели 2: 3 скрытых узла...")
model2 = MLPRegressor(
    hidden_layer_sizes=(3,),
    activation='logistic',  # сигмоида
    solver='adam',
    max_iter=1000,
    random_state=42,
    tol=1e-6
)
model2.fit(X_train, y_train.ravel())

# Предсказания
y_pred2 = model2.predict(X_test).reshape(-1, 1)
correlation2 = np.corrcoef(y_test.flatten(), y_pred2.flatten())[0, 1]

results['model2'] = {
    'weights': model2.coefs_,
    'intercepts': model2.intercepts_,
    'correlation': correlation2
}

print(f"Корреляция модели 2: {correlation2:.6f}")
print(f"Количество итераций: {model2.n_iter_}")
print(f"Потеря: {model2.loss_:.6f}")

# Модель 3: Три скрытых узла с функцией активации ReLU (близка к smReLu для больших значений)
print("\nОбучение модели 3: 3 скрытых узла с функцией активации ReLU...")
print("Примечание: ReLU используется как приближение smReLu (log(1+exp(x)))")
model3 = MLPRegressor(
    hidden_layer_sizes=(3,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    tol=1e-6
)
model3.fit(X_train, y_train.ravel())

# Предсказания
y_pred3 = model3.predict(X_test).reshape(-1, 1)
correlation3 = np.corrcoef(y_test.flatten(), y_pred3.flatten())[0, 1]

results['model3'] = {
    'weights': model3.coefs_,
    'intercepts': model3.intercepts_,
    'correlation': correlation3
}

print(f"Корреляция модели 3: {correlation3:.6f}")
print(f"Количество итераций: {model3.n_iter_}")
print(f"Потеря: {model3.loss_:.6f}")

# График зависимости предсказанных и фактических значений
def plot_predictions_vs_actual(y_test, y_pred, filename, title):
    """График предсказанных vs фактических значений"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color='black', s=20)
    
    # Линия идеального предсказания
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    
    ax.set_xlabel('Фактические значения (нормализованные)', fontsize=12)
    ax.set_ylabel('Предсказанные значения (нормализованные)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

plot_predictions_vs_actual(y_test, y_pred1, 'predictions_model1.png', 
                          f'Модель 1: Корреляция = {correlation1:.3f}')
plot_predictions_vs_actual(y_test, y_pred2, 'predictions_model2.png', 
                          f'Модель 2: Корреляция = {correlation2:.3f}')
plot_predictions_vs_actual(y_test, y_pred3, 'predictions_model3.png', 
                          f'Модель 3: Корреляция = {correlation3:.3f}')

# Сохранение результатов для отчета
results['test_data'] = {
    'y_test': y_test,
    'y_pred1': y_pred1,
    'y_pred2': y_pred2,
    'y_pred3': y_pred3,
    'X_test': X_test
}

results['correlations'] = {
    'model1': correlation1,
    'model2': correlation2,
    'model3': correlation3
}

# Денормализация данных
y_test_denorm = denormalize_data(y_test, scaler_y)
y_pred1_denorm = denormalize_data(y_pred1, scaler_y)
y_pred2_denorm = denormalize_data(y_pred2, scaler_y)
y_pred3_denorm = denormalize_data(y_pred3, scaler_y)

# Создание таблицы с результатами
comparison_df = pd.DataFrame({
    'Фактическая прочность': y_test_denorm.flatten()[:9],
    'Предсказанная (модель 1)': y_pred1_denorm.flatten()[:9],
    'Предсказанная (модель 2)': y_pred2_denorm.flatten()[:9],
    'Предсказанная (модель 3)': y_pred3_denorm.flatten()[:9]
})

comparison_df.to_csv('predictions_comparison.csv', index=False, encoding='utf-8-sig')

print("\nВсе вычисления завершены!")
print(f"Корреляция модель 1: {correlation1:.6f}")
print(f"Корреляция модель 2: {correlation2:.6f}")
print(f"Корреляция модель 3: {correlation3:.6f}")

# Сохранение результатов в pickle для генератора отчета
import pickle
with open('neural_network_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nРезультаты сохранены в neural_network_results.pkl")
