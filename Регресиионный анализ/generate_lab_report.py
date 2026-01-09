import os
import sys
import pickle
import pandas as pd
import base64
import numpy as np

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Загрузка результатов
print("Загрузка результатов...")
with open('regression_analysis_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Загрузка данных
data = results['data']

# Преобразование изображений в base64
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

# Подготовка значений для подстановки
n_samples = len(data)
r2 = results['r2']
adj_r2 = results['adj_r2']
mse = results['mse']
rmse = results['rmse']
f_statistic = results['f_statistic']
f_p_value = results['f_p_value']
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

# Статистика по данным
stats_desc = data.describe()
mean_y = stats_desc.loc['mean', 'Y']
mean_x1 = stats_desc.loc['mean', 'X1']
mean_x2 = stats_desc.loc['mean', 'X2']
mean_x3 = stats_desc.loc['mean', 'X3']
mean_x4 = stats_desc.loc['mean', 'X4']
min_y = stats_desc.loc['min', 'Y']
max_y = stats_desc.loc['max', 'Y']
min_x1 = stats_desc.loc['min', 'X1']
max_x1 = stats_desc.loc['max', 'X1']

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа: Регрессионный анализ</title>
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
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="author">
        <p><strong>Выполнил:</strong> Тимошинов Егор Борисович</p>
        <p><strong>Группа:</strong> 16</p>
    </div>
    
    <h1>Лабораторная работа</h1>
    <h2>Регрессионный анализ с помощью R</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов построения моделей множественной линейной регрессии и оценки их качества на основе статистических данных.</p>
    
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
    <p>В ходе выполнения лабораторной работы была построена модель множественной линейной регрессии для анализа зависимости переменной $Y$ от предикторов $X_1, X_2, X_3, X_4$. Модель показала {'хорошее' if r2 > 0.7 else 'удовлетворительное' if r2 > 0.5 else 'низкое'} качество с коэффициентом детерминации $R^2 = {r2:.6f}$, что означает, что модель объясняет {r2*100:.2f}% вариации зависимой переменной. F-статистика {'подтверждает' if f_p_value < 0.05 else 'не подтверждает'} статистическую значимость модели. Анализ остатков {'не выявил' if abs(results['residuals'].mean()) < 0.1 else 'выявил'} серьезных нарушений предположений регрессионного анализа.</p>
    
</body>
</html>"""

# Сохранение HTML файла
with open('Timoshinov_E_B_Lab_RegressionAnalysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML отчет успешно создан: Timoshinov_E_B_Lab_RegressionAnalysis.html")
