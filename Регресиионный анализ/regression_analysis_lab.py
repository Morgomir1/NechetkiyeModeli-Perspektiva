import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Загрузка данных
print("Загрузка данных...")
data = pd.read_csv('К-ПЗ-множ-регр.csv', encoding='utf-8', quotechar='"')
# Преобразование числовых столбцов
for col in ['Y', 'X1', 'X2', 'X3', 'X4']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Переименование столбцов для удобства (удаление кавычек если есть)
print("\nПервые строки данных:")
print(data.head())

print("\nОписательная статистика:")
print(data.describe())

# Подготовка данных
X = data[['X1', 'X2', 'X3', 'X4']].values
y = data['Y'].values

# Построение модели множественной регрессии
print("\nПостроение модели множественной регрессии...")
model = LinearRegression()
model.fit(X, y)

# Предсказания
y_pred = model.predict(X)

# Вычисление метрик
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
n = len(y)
p = X.shape[1]  # количество предикторов

# Вычисление скорректированного R²
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Остатки
residuals = y - y_pred

# Статистика для коэффициентов
X_with_intercept = np.column_stack([np.ones(n), X])
cov_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * mse
se_coef = np.sqrt(np.diag(cov_matrix))
t_values = np.concatenate([[model.intercept_], model.coef_]) / se_coef
p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - p - 1))

# F-статистика
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
f_statistic = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
f_p_value = 1 - stats.f.cdf(f_statistic, p, n - p - 1)

print("\nРезультаты регрессии:")
print(f"R² = {r2:.6f}")
print(f"Скорректированный R² = {adj_r2:.6f}")
print(f"MSE = {mse:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"F-статистика = {f_statistic:.6f}")
print(f"p-value (F) = {f_p_value:.6f}")

print("\nКоэффициенты модели:")
print(f"Intercept (β₀): {model.intercept_:.6f}")
for i, coef in enumerate(model.coef_):
    print(f"β{i+1} (X{i+1}): {coef:.6f}, t = {t_values[i+1]:.6f}, p = {p_values[i+1]:.6f}")

# Сохранение результатов
results = {
    'model': model,
    'r2': r2,
    'adj_r2': adj_r2,
    'mse': mse,
    'rmse': rmse,
    'f_statistic': f_statistic,
    'f_p_value': f_p_value,
    'coefficients': {
        'intercept': model.intercept_,
        'X1': model.coef_[0],
        'X2': model.coef_[1],
        'X3': model.coef_[2],
        'X4': model.coef_[3]
    },
    't_values': {
        'intercept': t_values[0],
        'X1': t_values[1],
        'X2': t_values[2],
        'X3': t_values[3],
        'X4': t_values[4]
    },
    'p_values': {
        'intercept': p_values[0],
        'X1': p_values[1],
        'X2': p_values[2],
        'X3': p_values[3],
        'X4': p_values[4]
    },
    'y': y,
    'y_pred': y_pred,
    'residuals': residuals,
    'data': data
}

# Сохранение результатов
import pickle
with open('regression_analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nРезультаты сохранены в regression_analysis_results.pkl")

# Построение графиков
print("\nПостроение графиков...")

# График 1: Фактические vs Предсказанные значения
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='#2E86AB', alpha=0.7, s=60)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Идеальная линия')
plt.xlabel('Фактические значения Y', fontsize=12)
plt.ylabel('Предсказанные значения Y', fontsize=12)
plt.title('Фактические vs Предсказанные значения', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Остатки vs Предсказанные значения
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='#A23B72', alpha=0.7, s=60)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Предсказанные значения Y', fontsize=12)
plt.ylabel('Остатки', fontsize=12)
plt.title('Остатки vs Предсказанные значения', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: Q-Q plot для остатков
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q график остатков (проверка нормальности)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('qq_plot_residuals.png', dpi=300, bbox_inches='tight')
plt.close()

# График 4: Зависимость Y от каждого предиктора
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
predictors = ['X1', 'X2', 'X3', 'X4']
colors = ['#F18F01', '#C73E1D', '#6A994E', '#BC4749']

for idx, (pred, color) in enumerate(zip(predictors, colors)):
    ax = axes[idx // 2, idx % 2]
    ax.scatter(data[pred], y, color=color, alpha=0.7, s=60, label='Данные')
    # Линия тренда
    z = np.polyfit(data[pred], y, 1)
    p = np.poly1d(z)
    ax.plot(data[pred], p(data[pred]), "r--", alpha=0.8, label='Тренд')
    ax.set_xlabel(pred, fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Зависимость Y от {pred}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('y_vs_predictors.png', dpi=300, bbox_inches='tight')
plt.close()

print("Графики сохранены.")
