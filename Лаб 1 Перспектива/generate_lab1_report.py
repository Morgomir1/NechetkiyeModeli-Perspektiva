# -*- coding: utf-8 -*-
"""
Генерация HTML отчета для лабораторной работы 1
"""
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
with open('lab1_results.pkl', 'rb') as f:
    results = pickle.load(f)

df = results['data']
summary_stats = results['summary_stats']
quantiles_20 = results['quantiles_20']
se_mean = results['se_mean']
ci_lower, ci_upper = results['confidence_interval_99']
cov_Y_X1 = results['cov_Y_X1']
corr_Y_X1 = results['corr_Y_X1']
correlation_matrix = results['correlation_matrix']
k_bins = results['k_bins']
h_binwidth = results['h_binwidth']

# Преобразование изображений в base64
def image_to_base64(image_path):
    """Конвертирует изображение в base64 строку"""
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')
    return None

# Загрузка изображений
histogram_Y_img = image_to_base64('histogram_Y.png')
histogram_Y_detailed_img = image_to_base64('histogram_Y_detailed.png')
boxplot_Y_img = image_to_base64('boxplot_Y.png')
boxplot_X1_img = image_to_base64('boxplot_X1.png')
boxplot_Y_horizontal_img = image_to_base64('boxplot_Y_horizontal.png')
scatter_X1_Y_img = image_to_base64('scatter_X1_Y.png')
pairs_plot_img = image_to_base64('pairs_plot.png')
correlation_heatmap_img = image_to_base64('correlation_heatmap.png')

# Создание таблицы с данными (первые 20 строк)
data_table_rows = []
for idx, row in df.head(20).iterrows():
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

# Таблица описательных статистик
stats_table_rows = []
for var in df.columns:
    stats_table_rows.append(f"""
        <tr>
            <td>{var}</td>
            <td>{df[var].mean():.4f}</td>
            <td>{df[var].median():.4f}</td>
            <td>{df[var].min():.4f}</td>
            <td>{df[var].max():.4f}</td>
            <td>{df[var].std():.4f}</td>
            <td>{df[var].var():.4f}</td>
        </tr>""")

stats_table = "\n".join(stats_table_rows)

# Таблица квантилей
quantiles_table_rows = []
quantile_labels = [f"{int(q*100)}%" for q in np.arange(0, 1.01, 0.20)]
for label, value in zip(quantile_labels, quantiles_20):
    quantiles_table_rows.append(f"""
        <tr>
            <td>{label}</td>
            <td>{value:.4f}</td>
        </tr>""")

quantiles_table = "\n".join(quantiles_table_rows)

# Таблица корреляционной матрицы
corr_table_rows = []
for i, col1 in enumerate(df.columns):
    row_cells = [f"<td><strong>{col1}</strong></td>"]
    for j, col2 in enumerate(df.columns):
        corr_value = correlation_matrix.loc[col1, col2]
        row_cells.append(f"<td>{corr_value:.4f}</td>")
    corr_table_rows.append(f"<tr>{''.join(row_cells)}</tr>")

corr_table = "\n".join(corr_table_rows)
corr_header = "<tr><th></th>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа 1: Перспективные информационные технологии</title>
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
        .task {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid black;
        }}
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 5px 0;
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
        <p><strong>Выполнил:</strong> Тимошинов Егор Борисович</p>
        <p><strong>Группа:</strong> 16</p>
    </div>
    
    <h1>Лабораторная работа 1</h1>
    <h2>Перспективные информационные технологии</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов описательного анализа данных, построения графиков и расчета статистических характеристик с использованием языка программирования Python.</p>
    
    <h3>Теоретические сведения</h3>
    <p>Описательная статистика позволяет получить общее представление о данных. Основные характеристики включают меры центральной тенденции (среднее, медиана), меры вариации (дисперсия, стандартное отклонение), а также меры формы распределения (асимметрия, эксцесс).</p>
    
    <p>Графические методы визуализации данных помогают наглядно представить структуру данных и выявить закономерности. К основным типам графиков относятся гистограммы, ящики с усами, диаграммы рассеяния.</p>
    
    <h2>Результаты выполнения задания</h2>
    
    <h3>1. Исходные данные</h3>
    <p>Для анализа использованы данные о {len(df)} наблюдениях. Переменные: $Y$ (зависимая переменная), $X_1, X_2, X_3, X_4$ (независимые переменные).</p>
    
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
    
    <h3>2. Структура данных</h3>
    <p>Количество наблюдений: {len(df)}</p>
    <p>Количество переменных: {len(df.columns)}</p>
    <p>Типы данных: все переменные являются числовыми (float64)</p>
    
    <h3>3. Гистограммы</h3>
    <p>Для построения гистограммы переменной $Y$ было рассчитано оптимальное количество интервалов по формуле Стерджеса:</p>
    <div class="formula">$k = 1 + 3.322 \\log_{{10}}(n) = {k_bins}$</div>
    <p>где $n = {len(df)}$ — количество наблюдений.</p>
    <p>Ширина интервала: $h = {h_binwidth:.4f}$</p>
    
    <div class="task">
        <img src="data:image/png;base64,{histogram_Y_img}" alt="Гистограмма частот">
        <div class="figure-caption">Рисунок 1. Гистограмма частот для переменной Y</div>
    </div>
    
    <div class="task">
        <img src="data:image/png;base64,{histogram_Y_detailed_img}" alt="Гистограмма с заданными границами">
        <div class="figure-caption">Рисунок 2. Гистограмма для переменной Y с заданными границами интервалов</div>
    </div>
    
    <h3>4. Описательные статистики</h3>
    <p>Основные описательные статистики для всех переменных:</p>
    <table>
        <tr>
            <th>Переменная</th>
            <th>Среднее</th>
            <th>Медиана</th>
            <th>Минимум</th>
            <th>Максимум</th>
            <th>Ст. отклонение</th>
            <th>Дисперсия</th>
        </tr>
        {stats_table}
    </table>
    
    <p>Дополнительные характеристики для переменной $Y$:</p>
    <ul>
        <li>Среднее арифметическое: {summary_stats['mean']:.4f}</li>
        <li>Медиана: {summary_stats['median']:.4f}</li>
        <li>Стандартное отклонение: {summary_stats['std']:.4f}</li>
        <li>Дисперсия: {summary_stats['var']:.4f}</li>
        <li>Размах: {summary_stats['range']:.4f}</li>
        <li>Коэффициент вариации: {summary_stats['cv']:.2f}%</li>
        <li>Коэффициент асимметрии: {summary_stats['skewness']:.4f}</li>
        <li>Коэффициент эксцесса: {summary_stats['kurtosis']:.4f}</li>
        <li>Стандартная ошибка среднего: {se_mean:.4f}</li>
    </ul>
    
    <p>Квантили переменной $Y$ (с шагом 0.20):</p>
    <table>
        <tr>
            <th>Квантиль</th>
            <th>Значение</th>
        </tr>
        {quantiles_table}
    </table>
    
    <p>99% доверительный интервал для среднего значения $Y$:</p>
    <div class="formula">$[{ci_lower:.4f}, {ci_upper:.4f}]$</div>
    
    <h3>5. Ящики с усами</h3>
    <p>Ящик с усами (box plot) позволяет визуализировать распределение данных, показать медиану, квартили и выбросы.</p>
    
    <div class="task">
        <img src="data:image/png;base64,{boxplot_Y_img}" alt="Ящик с усами для Y">
        <div class="figure-caption">Рисунок 3. Ящик с усами для переменной Y</div>
    </div>
    
    <div class="task">
        <img src="data:image/png;base64,{boxplot_X1_img}" alt="Ящик с усами для X1">
        <div class="figure-caption">Рисунок 4. Ящик с усами для переменной X1</div>
    </div>
    
    <div class="task">
        <img src="data:image/png;base64,{boxplot_Y_horizontal_img}" alt="Горизонтальный ящик с усами для Y">
        <div class="figure-caption">Рисунок 5. Горизонтальный ящик с усами для переменной Y с отображением точек</div>
    </div>
    
    <h3>6. Диаграммы рассеяния</h3>
    <p>Диаграммы рассеяния позволяют визуализировать связь между переменными.</p>
    
    <div class="task">
        <img src="data:image/png;base64,{scatter_X1_Y_img}" alt="Диаграмма рассеяния Y от X1">
        <div class="figure-caption">Рисунок 6. Диаграмма рассеяния для переменных X1 и Y</div>
    </div>
    
    <div class="task">
        <img src="data:image/png;base64,{pairs_plot_img}" alt="Матрица диаграмм рассеяния">
        <div class="figure-caption">Рисунок 7. Матрица диаграмм рассеяния для всех переменных</div>
    </div>
    
    <h3>7. Ковариация и корреляция</h3>
    <p>Ковариация между переменными $Y$ и $X_1$:</p>
    <div class="formula">$\\text{{cov}}(Y, X_1) = {cov_Y_X1:.4f}$</div>
    
    <p>Коэффициент корреляции между переменными $Y$ и $X_1$:</p>
    <div class="formula">$r_{{Y,X_1}} = {corr_Y_X1:.4f}$</div>
    
    <p>Корреляционная матрица для всех переменных:</p>
    <table>
        {corr_header}
        {corr_table}
    </table>
    
    <div class="task">
        <img src="data:image/png;base64,{correlation_heatmap_img}" alt="Тепловая карта корреляционной матрицы">
        <div class="figure-caption">Рисунок 8. Тепловая карта корреляционной матрицы</div>
    </div>
    
    <h3>Выводы</h3>
    <p>В ходе выполнения лабораторной работы был проведен описательный анализ данных. Построены гистограммы, ящики с усами и диаграммы рассеяния для визуализации распределений и связей между переменными. Рассчитаны основные описательные статистики, включая меры центральной тенденции, вариации и формы распределения. Вычислена корреляционная матрица, которая показывает степень линейной связи между переменными. Коэффициент корреляции между $Y$ и $X_1$ составляет {corr_Y_X1:.4f}, что указывает на {'сильную' if abs(corr_Y_X1) > 0.7 else 'умеренную' if abs(corr_Y_X1) > 0.3 else 'слабую'} {'положительную' if corr_Y_X1 > 0 else 'отрицательную'} связь между этими переменными.</p>
    
</body>
</html>"""

# Сохранение HTML файла
with open('Timoshinov_E_B_Lab_1_Perspective.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML отчет успешно создан: Timoshinov_E_B_Lab_1_Perspective.html")
