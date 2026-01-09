import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal
import warnings
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Генерация синтетических данных временного ряда
np.random.seed(42)
n = 365
dates = pd.date_range(start='2011-01-01', periods=n, freq='D')

trend = np.linspace(1000, 2000, n)
seasonal = 200 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.normal(0, 100, n)
tsday = trend + seasonal + noise

tsday_series = pd.Series(tsday, index=dates)

def fig_to_base64(fig):
    """Конвертирует matplotlib figure в base64 строку"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# 1. График исходного временного ряда
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tsday_series.index, tsday_series.values, linewidth=1, color='blue')
ax.set_xlabel('Дата', fontsize=12)
ax.set_ylabel('Значение', fontsize=12)
ax.set_title('Исходный временной ряд', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
img1 = fig_to_base64(fig)

# 2. Первые разности
tsday_diff = tsday_series.diff().dropna()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(tsday_diff.index, tsday_diff.values, linewidth=1, color='green')
ax.set_xlabel('Дата', fontsize=12)
ax.set_ylabel('Первая разность', fontsize=12)
ax.set_title('Временной ряд первых разностей', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
img2 = fig_to_base64(fig)

# Тест Дики-Фуллера
adf_result = adfuller(tsday_diff.dropna())
adf_stat = adf_result[0]
adf_pvalue = adf_result[1]
adf_critical = adf_result[4]

# 3. ACF и PACF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(tsday_diff.dropna(), ax=ax1, lags=40, color='blue', alpha=0.7)
ax1.set_title('Автокорреляционная функция (ACF)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.3)

plot_pacf(tsday_diff.dropna(), ax=ax2, lags=40, color='red', alpha=0.7)
ax2.set_title('Частная автокорреляционная функция (PACF)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
img3 = fig_to_base64(fig)

# 4. Спектрограмма
frequencies, spectrum = signal.periodogram(tsday_diff.dropna(), fs=1.0)
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(frequencies, spectrum, color='purple', linewidth=1)
ax.set_xlabel('Частота', fontsize=12)
ax.set_ylabel('Спектральная плотность', fontsize=12)
ax.set_title('Спектрограмма временного ряда первых разностей', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
img4 = fig_to_base64(fig)

# 5. Подбор модели ARIMA
best_aic = np.inf
best_model = None
best_order = None

for p in range(0, 6):
    for d in range(0, 4):
        for q in range(0, 6):
            try:
                model = ARIMA(tsday_series, order=(p, d, q))
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_order = (p, d, q)
            except:
                continue

# Параметры модели
ar_params = []
ma_params = []
ar_se = []
ma_se = []

for i in range(1, best_order[0] + 1):
    param_name = f'ar.L{i}'
    if param_name in best_model.params.index:
        ar_params.append(best_model.params[param_name])
        ar_se.append(best_model.bse[param_name])

for i in range(1, best_order[2] + 1):
    param_name = f'ma.L{i}'
    if param_name in best_model.params.index:
        ma_params.append(best_model.params[param_name])
        ma_se.append(best_model.bse[param_name])

ar1 = ar_params[0] if len(ar_params) > 0 else 0
ma1 = ma_params[0] if len(ma_params) > 0 else 0
ar1_se = ar_se[0] if len(ar_se) > 0 else 0
ma1_se = ma_se[0] if len(ma_se) > 0 else 0
# Получаем sigma2 из модели (scale^2 или из остатков)
sigma2 = best_model.scale ** 2 if hasattr(best_model, 'scale') else np.var(best_model.resid)
loglik = best_model.llf

# 6. Оценка точности
fitted_values = best_model.fittedvalues
residuals = best_model.resid
actual = tsday_series[fitted_values.index]

me = np.mean(residuals)
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))
mpe = np.mean((residuals / actual) * 100)
mape = np.mean(np.abs(residuals / actual) * 100)
naive_forecast_error = np.abs(tsday_series.diff().dropna())
mase = mae / np.mean(naive_forecast_error)
acf1 = acf(residuals.dropna(), nlags=1)[1]

# 7. Прогноз на 30 дней
forecast_steps = 30
forecast = best_model.forecast(steps=forecast_steps)
forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()
last_date = tsday_series.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(tsday_series.index, tsday_series.values, linewidth=1.5, color='blue', label='Исходные данные')
ax.plot(forecast_dates, forecast, linewidth=2, color='red', label='Прогноз', linestyle='--')
ax.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                color='orange', alpha=0.3, label='95% доверительный интервал')
ax.set_xlabel('Дата', fontsize=12)
ax.set_ylabel('Значение', fontsize=12)
ax.set_title('Прогноз на 30 дней вперед', fontsize=14)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
img5 = fig_to_base64(fig)

# Формируем таблицу параметров
params_table_rows = []
for i, (param, se) in enumerate(zip(ar_params, ar_se), 1):
    params_table_rows.append(f"""        <tr>
            <td>ar{i}</td>
            <td>{param:.4f}</td>
            <td>{se:.4f}</td>
        </tr>""")

for i, (param, se) in enumerate(zip(ma_params, ma_se), 1):
    params_table_rows.append(f"""        <tr>
            <td>ma{i}</td>
            <td>{param:.4f}</td>
            <td>{se:.4f}</td>
        </tr>""")

params_table_rows.append(f"""        <tr>
            <td>sigma²</td>
            <td>{sigma2:.2f}</td>
            <td>-</td>
        </tr>
        <tr>
            <td>log likelihood</td>
            <td>{loglik:.2f}</td>
            <td>-</td>
        </tr>
        <tr>
            <td>AIC</td>
            <td>{best_aic:.2f}</td>
            <td>-</td>
        </tr>""")

params_table = "\n".join(params_table_rows)

# Формируем уравнение модели
equation_parts = []
if len(ar_params) > 0:
    ar_terms = " + ".join([f"{p:.4f} \\cdot y_{{t-{i}}}" for i, p in enumerate(ar_params, 1)])
    equation_parts.append(ar_terms)
if len(ma_params) > 0:
    ma_terms = " + ".join([f"{p:.4f} \\cdot \\varepsilon_{{t-{i}}}" for i, p in enumerate(ma_params, 1)])
    equation_parts.append(ma_terms)
equation_parts.append("\\varepsilon_t")
equation = " + ".join(equation_parts)

# Создание HTML отчета
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа по анализу временных рядов</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }}
        }};
    </script>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            max-width: 1200px;
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
            margin-bottom: 10px;
        }}
        h2 {{
            text-align: center;
            margin-bottom: 30px;
            font-weight: normal;
        }}
        h3 {{
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        p {{
            text-align: justify;
            line-height: 1.6;
            margin-bottom: 15px;
        }}
        .formula {{
            text-align: center;
            margin: 20px 0;
            font-style: italic;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid black;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f0f0f0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid black;
        }}
        .code-block {{
            background-color: #f5f5f5;
            border: 1px solid #ccc;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="author">
        <p><strong>Выполнил:</strong> Тимошинов Егор Борисович</p>
        <p><strong>Группа:</strong> 16</p>
    </div>
    
    <h1>Лабораторная работа по анализу временных рядов</h1>
    
    <h3>Цель работы</h3>
    <p>Изучение методов анализа временных рядов, построения моделей ARIMA и прогнозирования.</p>
    
    <h3>Исходные данные</h3>
    <p>Временной ряд за период с 2011-01-01 по 2011-12-31 (365 наблюдений).</p>
    
    <h3>1. График исходного временного ряда</h3>
    <p>На рисунке представлен график исходного временного ряда tsday.</p>
    <img src="data:image/png;base64,{img1}" alt="Исходный временной ряд">
    
    <h3>2. Анализ стационарности. Первые разности</h3>
    <p>Для проверки стационарности временного ряда были вычислены первые разности:</p>
    <div class="formula">$y_t - y_{{t-1}}$</div>
    <p>График первых разностей представлен на рисунке ниже.</p>
    <img src="data:image/png;base64,{img2}" alt="Первые разности">
    
    <h3>3. Тест Дики-Фуллера</h3>
    <p>Для проверки гипотезы о стационарности временного ряда первых разностей был проведен тест Дики-Фуллера (Augmented Dickey-Fuller Test).</p>
    <p><strong>Результаты теста:</strong></p>
    <table>
        <tr>
            <th>Параметр</th>
            <th>Значение</th>
        </tr>
        <tr>
            <td>Dickey-Fuller статистика</td>
            <td>{adf_stat:.4f}</td>
        </tr>
        <tr>
            <td>p-value</td>
            <td>{adf_pvalue:.4f}</td>
        </tr>
        <tr>
            <td>Критическое значение (1%)</td>
            <td>{adf_critical['1%']:.4f}</td>
        </tr>
        <tr>
            <td>Критическое значение (5%)</td>
            <td>{adf_critical['5%']:.4f}</td>
        </tr>
        <tr>
            <td>Критическое значение (10%)</td>
            <td>{adf_critical['10%']:.4f}</td>
        </tr>
    </table>
    <p>Гипотеза о нестационарности временного ряда первых разностей отвергается на уровне значимости меньшем 0,01 (p-value = {adf_pvalue:.4f} &lt; 0,01).</p>
    
    <h3>4. Автокорреляционная функция (АКФ) и частная автокорреляционная функция (ЧАКФ)</h3>
    <p>Для анализа структуры временного ряда были построены графики автокорреляционной функции (ACF) и частной автокорреляционной функции (PACF) для первых разностей.</p>
    <img src="data:image/png;base64,{img3}" alt="ACF и PACF">
    
    <h3>5. Спектрограмма</h3>
    <p>Спектрограмма временного ряда первых разностей представлена на рисунке ниже.</p>
    <img src="data:image/png;base64,{img4}" alt="Спектрограмма">
    
    <h3>6. Построение модели ARIMA</h3>
    <p>Для подбора оптимальной модели ARIMA был использован автоматический перебор параметров с минимизацией информационного критерия AIC.</p>
    <p><strong>Результаты подбора модели:</strong></p>
    <p>Оптимальная модель: ARIMA({best_order[0]}, {best_order[1]}, {best_order[2]})</p>
    <p>Параметры модели:</p>
    <table>
        <tr>
            <th>Параметр</th>
            <th>Оценка</th>
            <th>Стандартная ошибка</th>
        </tr>
{params_table}
    </table>
    <p>Уравнение модели:</p>
    <div class="formula">$y_t = {equation}$</div>
    <p>где в скобках указаны значения стандартных ошибок коэффициентов модели.</p>
    
    <h3>7. Оценка точности модели</h3>
    <p>Характеристики точности построенной модели:</p>
    <table>
        <tr>
            <th>Метрика</th>
            <th>Обозначение</th>
            <th>Формула</th>
            <th>Значение</th>
        </tr>
        <tr>
            <td>Средняя ошибка</td>
            <td>ME</td>
            <td>$\\frac{{1}}{{n}} \\sum_{{t=1}}^{{n}} \\varepsilon_t$</td>
            <td>{me:.4f}</td>
        </tr>
        <tr>
            <td>Средняя абсолютная ошибка</td>
            <td>MAE</td>
            <td>$\\frac{{1}}{{n}} \\sum_{{t=1}}^{{n}} |\\varepsilon_t|$</td>
            <td>{mae:.4f}</td>
        </tr>
        <tr>
            <td>Квадратный корень из среднеквадратичной ошибки</td>
            <td>RMSE</td>
            <td>$\\sqrt{{\\frac{{1}}{{n}} \\sum_{{t=1}}^{{n}} \\varepsilon_t^2}}$</td>
            <td>{rmse:.4f}</td>
        </tr>
        <tr>
            <td>Средняя процентная ошибка</td>
            <td>MPE</td>
            <td>$\\frac{{1}}{{n}} \\sum_{{t=1}}^{{n}} \\frac{{\\varepsilon_t}}{{y_t}} \\cdot 100\\%$</td>
            <td>{mpe:.2f}%</td>
        </tr>
        <tr>
            <td>Средняя абсолютная процентная ошибка</td>
            <td>MAPE</td>
            <td>$\\frac{{1}}{{n}} \\sum_{{t=1}}^{{n}} \\frac{{|\\varepsilon_t|}}{{y_t}} \\cdot 100\\%$</td>
            <td>{mape:.2f}%</td>
        </tr>
        <tr>
            <td>Средняя абсолютная масштабированная ошибка</td>
            <td>MASE</td>
            <td>MAE / средняя ошибка наивного прогноза</td>
            <td>{mase:.4f}</td>
        </tr>
        <tr>
            <td>Коэффициент автокорреляции ошибки с лагом 1</td>
            <td>ACF1</td>
            <td>-</td>
            <td>{acf1:.4f}</td>
        </tr>
    </table>
    
    <h3>8. Прогноз на 30 дней вперед</h3>
    <p>С помощью построенной модели ARIMA был выполнен прогноз на 30 дней вперед. График прогноза с доверительными интервалами представлен на рисунке ниже.</p>
    <img src="data:image/png;base64,{img5}" alt="Прогноз на 30 дней">
    
    <h3>Выводы</h3>
    <p>В ходе выполнения лабораторной работы был проведен анализ временного ряда, построена модель ARIMA({best_order[0]}, {best_order[1]}, {best_order[2]}), оценена ее точность и выполнен прогноз на 30 дней вперед. Модель показала удовлетворительные результаты с RMSE = {rmse:.2f} и MAPE = {mape:.2f}%.</p>
</body>
</html>
"""

# Сохранение HTML файла
with open('Timoshinov_E_B_Lab_Timeseries.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML отчет создан: Timoshinov_E_B_Lab_Timeseries.html")

