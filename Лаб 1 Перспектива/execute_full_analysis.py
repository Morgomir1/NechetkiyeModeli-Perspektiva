# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
import pickle
import base64
warnings.filterwarnings('ignore')

# Меняем рабочую директорию на директорию скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("=" * 60)
print("Лабораторная работа: Множественная регрессия")
print("Выполнил: Тимошинов Егор Борисович, группа 16")
print("=" * 60)

# Загрузка данных
print("\n[1/4] Загрузка данных...")
data = pd.read_csv('К ПЗ множ регр.csv', sep=';', decimal=',')

print(f"Загружено {len(data)} наблюдений")
print("\nПервые строки данных:")
print(data.head())

# Подготовка данных
X = data[['X1', 'X2', 'X3', 'X4']].values
y = data['Y'].values

# Построение модели множественной регрессии
print("\n[2/4] Построение модели множественной регрессии...")
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

with open('multiple_regression_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nРезультаты сохранены в multiple_regression_results.pkl")

# Построение графиков
print("\n[3/4] Построение графиков...")

# График 1: Фактические vs Предсказанные значения
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.6)
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
plt.scatter(y_pred, residuals, color='green', alpha=0.6)
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
colors = ['red', 'blue', 'green', 'orange']

for idx, (pred, color) in enumerate(zip(predictors, colors)):
    ax = axes[idx // 2, idx % 2]
    ax.scatter(data[pred], y, color=color, alpha=0.6, label='Данные')
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

# Генерация HTML отчета
print("\n[4/4] Генерация HTML отчета...")

def image_to_base64(image_path):
    """Конвертирует изображение в base64 строку"""
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')
    return None

# Загрузка изображений
actual_vs_predicted_img = image_to_base64('actual_vs_predicted.png')
residuals_vs_predicted_img = image_to_base64('residuals_vs_predicted.png')
qq_plot_img = image_to_base64('qq_plot_residuals.png')
y_vs_predictors_img = image_to_base64('y_vs_predictors.png')

# Создание таблицы с данными (первые 20 строк)
data_table_rows = []
for idx, row in data.head(20).iterrows():
    data_table_rows.append(f"""
        <tr>
            <td>{idx + 1}</td>
            <td>{row['Y']:.1f}</td>
            <td>{row['X1']:.1f}</td>
            <td>{row['X2']:.1f}</td>
            <td>{row['X3']:.1f}</td>
            <td>{row['X4']:.1f}</td>
        </tr>""")

data_table = "\n".join(data_table_rows)

# Статистика по данным
n_samples = n  # Количество наблюдений
stats_desc = data.describe()
mean_y = stats_desc.loc['mean', 'Y']
mean_x1 = stats_desc.loc['mean', 'X1']
mean_x2 = stats_desc.loc['mean', 'X2']
mean_x3 = stats_desc.loc['mean', 'X3']
mean_x4 = stats_desc.loc['mean', 'X4']
min_y = stats_desc.loc['min', 'Y']
max_y = stats_desc.loc['max', 'Y']

# Извлечение коэффициентов для HTML
intercept = results['coefficients']['intercept']
coef_x1 = results['coefficients']['X1']
coef_x2 = results['coefficients']['X2']
coef_x3 = results['coefficients']['X3']
coef_x4 = results['coefficients']['X4']
t_intercept = results['t_values']['intercept']
t_x1 = results['t_values']['X1']
t_x2 = results['t_values']['X2']
t_x3 = results['t_values']['X3']
t_x4 = results['t_values']['X4']
p_intercept = results['p_values']['intercept']
p_x1 = results['p_values']['X1']
p_x2 = results['p_values']['X2']
p_x3 = results['p_values']['X3']
p_x4 = results['p_values']['X4']

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа: Множественная регрессия</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: black;
            background-color: white;
        }}
        .author {{
            text-align: right;
            margin-bottom: 20px;
            font-size: 14pt;
        }}
        h1 {{
            text-align: center;
            font-size: 18pt;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        h2 {{
            font-size: 16pt;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        h3 {{
            font-size: 14pt;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        p {{
            text-align: justify;
            line-height: 1.5;
            margin: 10px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        table, th, td {{
            border: 1px solid black;
        }}
        th, td {{
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: white;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
            border: 1px solid black;
        }}
        .figure-caption {{
            text-align: center;
            font-style: italic;
            margin-top: 5px;
            margin-bottom: 15px;
        }}
        .formula {{
            text-align: center;
            margin: 15px 0;
            font-style: italic;
        }}
        .code-block {{
            background-color: #f5f5f5;
            border: 1px solid #ccc;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            overflow-x: auto;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="author">
        Тимошинов Егор Борисович, группа 16
    </div>
    
    <h1>Лабораторная работа</h1>
    <h2>Множественная регрессия</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов построения моделей множественной линейной регрессии и оценки их качества.</p>
    
    <h3>Теоретические сведения</h3>
    <p>Множественная линейная регрессия используется для моделирования зависимости одной зависимой переменной от нескольких независимых переменных. Модель позволяет оценить влияние каждого предиктора на зависимую переменную с учетом влияния других переменных.</p>
    
    <p>Модель множественной линейной регрессии имеет вид:</p>
    <div class="formula">
        $Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\beta_3 X_3 + \\beta_4 X_4 + \\varepsilon$
    </div>
    
    <p>где $Y$ — зависимая переменная, $X_1, X_2, X_3, X_4$ — независимые переменные (предикторы), $\\beta_0$ — свободный член, $\\beta_1, \\beta_2, \\beta_3, \\beta_4$ — коэффициенты регрессии, $\\varepsilon$ — случайная ошибка.</p>
    
    <h3>Задание</h3>
    <ol>
        <li>Загрузить данные и провести их описательный анализ.</li>
        <li>Построить модель множественной линейной регрессии.</li>
        <li>Оценить значимость коэффициентов регрессии.</li>
        <li>Оценить качество модели с помощью коэффициента детерминации и F-статистики.</li>
        <li>Проанализировать остатки модели.</li>
        <li>Построить графики для визуализации результатов.</li>
    </ol>
    
    <h2>Результаты выполнения задания</h2>
    
    <h3>Исходные данные</h3>
    <p>Для анализа использованы данные о {n_samples} наблюдениях. Зависимая переменная — $Y$, независимые переменные — $X_1, X_2, X_3, X_4$.</p>
    
    <p>Описательная статистика по данным:</p>
    <ul>
        <li>Среднее значение $Y$: {mean_y:.2f}</li>
        <li>Минимальное значение $Y$: {min_y:.1f}</li>
        <li>Максимальное значение $Y$: {max_y:.1f}</li>
        <li>Среднее значение $X_1$: {mean_x1:.2f}</li>
        <li>Среднее значение $X_2$: {mean_x2:.2f}</li>
        <li>Среднее значение $X_3$: {mean_x3:.2f}</li>
        <li>Среднее значение $X_4$: {mean_x4:.2f}</li>
    </ul>
    
    <p>Первые 20 наблюдений:</p>
    <table>
        <tr>
            <th>№</th>
            <th>Y</th>
            <th>X1</th>
            <th>X2</th>
            <th>X3</th>
            <th>X4</th>
        </tr>
        {data_table}
    </table>
    
    <h3>Результаты моделирования</h3>
    <p>Коэффициенты модели множественной линейной регрессии:</p>
    <table>
        <tr>
            <th>Коэффициент</th>
            <th>Значение</th>
            <th>t-статистика</th>
            <th>p-value</th>
        </tr>
        <tr>
            <td>$\\beta_0$ (Intercept)</td>
            <td>{intercept:.6f}</td>
            <td>{t_intercept:.6f}</td>
            <td>{p_intercept:.6f}</td>
        </tr>
        <tr>
            <td>$\\beta_1$ (X1)</td>
            <td>{coef_x1:.6f}</td>
            <td>{t_x1:.6f}</td>
            <td>{p_x1:.6f}</td>
        </tr>
        <tr>
            <td>$\\beta_2$ (X2)</td>
            <td>{coef_x2:.6f}</td>
            <td>{t_x2:.6f}</td>
            <td>{p_x2:.6f}</td>
        </tr>
        <tr>
            <td>$\\beta_3$ (X3)</td>
            <td>{coef_x3:.6f}</td>
            <td>{t_x3:.6f}</td>
            <td>{p_x3:.6f}</td>
        </tr>
        <tr>
            <td>$\\beta_4$ (X4)</td>
            <td>{coef_x4:.6f}</td>
            <td>{t_x4:.6f}</td>
            <td>{p_x4:.6f}</td>
        </tr>
    </table>
    
    <p>Уравнение модели:</p>
    <div class="formula">
        $Y = {intercept:.6f} + {coef_x1:.6f} \\cdot X_1 + {coef_x2:.6f} \\cdot X_2 + {coef_x3:.6f} \\cdot X_3 + {coef_x4:.6f} \\cdot X_4$
    </div>
    
    <h3>Оценка качества модели</h3>
    <p>Метрики качества модели:</p>
    <ul>
        <li>Коэффициент детерминации ($R^2$): {r2:.6f}</li>
        <li>Скорректированный $R^2$: {adj_r2:.6f}</li>
        <li>Среднеквадратическая ошибка (MSE): {mse:.6f}</li>
        <li>Корень из среднеквадратической ошибки (RMSE): {rmse:.6f}</li>
        <li>F-статистика: {f_statistic:.6f}</li>
        <li>p-value (F): {f_p_value:.6f}</li>
    </ul>
    
    <p>Коэффициент детерминации $R^2 = {r2:.6f}$ показывает, что модель объясняет {r2*100:.2f}% вариации зависимой переменной. F-статистика = {f_statistic:.6f} с p-value = {f_p_value:.6f} {'указывает на статистическую значимость модели' if f_p_value < 0.05 else 'не указывает на статистическую значимость модели'}.</p>
    
    <div class="task">
        <h3>График фактических vs предсказанных значений</h3>
        <img src="data:image/png;base64,{actual_vs_predicted_img}" alt="Фактические vs Предсказанные значения">
        <div class="figure-caption">Рисунок 1. Сравнение фактических и предсказанных значений</div>
        <p>На графике показано сравнение фактических значений зависимой переменной $Y$ с предсказанными значениями модели. Точки, близкие к диагональной линии, указывают на хорошее качество модели.</p>
    </div>
    
    <div class="task">
        <h3>График остатков vs предсказанные значения</h3>
        <img src="data:image/png;base64,{residuals_vs_predicted_img}" alt="Остатки vs Предсказанные значения">
        <div class="figure-caption">Рисунок 2. Зависимость остатков от предсказанных значений</div>
        <p>График остатков позволяет проверить предположение о гомоскедастичности (постоянстве дисперсии ошибок). Остатки должны быть случайно распределены вокруг нуля без видимых закономерностей.</p>
    </div>
    
    <div class="task">
        <h3>Q-Q график остатков</h3>
        <img src="data:image/png;base64,{qq_plot_img}" alt="Q-Q график остатков">
        <div class="figure-caption">Рисунок 3. Q-Q график остатков для проверки нормальности распределения</div>
        <p>Q-Q график используется для проверки нормальности распределения остатков. Если остатки нормально распределены, точки должны лежать близко к прямой линии.</p>
    </div>
    
    <div class="task">
        <h3>Зависимость Y от предикторов</h3>
        <img src="data:image/png;base64,{y_vs_predictors_img}" alt="Зависимость Y от предикторов">
        <div class="figure-caption">Рисунок 4. Зависимость зависимой переменной Y от каждого предиктора</div>
        <p>На графиках показана зависимость $Y$ от каждого из предикторов $X_1, X_2, X_3, X_4$. Линия тренда показывает общую направленность связи между переменными.</p>
    </div>
    
    <h3>Выводы</h3>
    <p>В ходе выполнения лабораторной работы была построена модель множественной линейной регрессии для анализа зависимости переменной $Y$ от предикторов $X_1, X_2, X_3, X_4$. Модель показала {'хорошее' if r2 > 0.7 else 'удовлетворительное' if r2 > 0.5 else 'низкое'} качество с коэффициентом детерминации $R^2 = {r2:.6f}$, что означает, что модель объясняет {r2*100:.2f}% вариации зависимой переменной. F-статистика {'подтверждает' if f_p_value < 0.05 else 'не подтверждает'} статистическую значимость модели. Анализ остатков {'не выявил' if abs(residuals.mean()) < 0.1 else 'выявил'} серьезных нарушений предположений регрессионного анализа.</p>
    
</body>
</html>"""

# Сохранение HTML файла
with open('Timoshinov_E_B_Lab_MultipleRegression.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML отчет успешно создан: Timoshinov_E_B_Lab_MultipleRegression.html")
print("\n" + "=" * 60)
print("Лабораторная работа выполнена успешно!")
print("=" * 60)
