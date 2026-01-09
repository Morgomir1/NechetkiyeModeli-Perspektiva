# -*- coding: utf-8 -*-
"""
Генерация HTML отчета для лабораторной работы 10
"""

import os
import pickle
import sys
import io
import numpy as np

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Загрузка результатов
print("Загрузка результатов...")
with open('fuzzy_comparison_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Загрузка изображений
try:
    with open('plot_images.pkl', 'rb') as f:
        images = pickle.load(f)
    print("Изображения загружены")
except Exception as e:
    print(f"Ошибка загрузки изображений: {e}")
    images = {}

def get_image(key, default=""):
    """Получить изображение в формате base64"""
    if key in images and images[key]:
        return f'<img src="data:image/png;base64,{images[key]}" alt="{key}" style="max-width: 100%; height: auto; display: block; margin: 15px auto; border: 1px solid black;">'
    return default

# Подготовка таблицы сравнения
comparison_table_rows = ""
for i, comp in enumerate(results['comparison'], 1):
    comparison_table_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{comp['input1']:.1f}</td>
            <td>{comp['input2']:.1f}</td>
            <td>{comp['mamdani']:.4f}</td>
            <td>{comp['sugeno']:.4f}</td>
            <td>{comp['difference']:.4f}</td>
        </tr>"""

# Статистика
mamdani_outputs = [r['mamdani'] for r in results['comparison']]
sugeno_outputs = [r['sugeno'] for r in results['comparison']]
differences = [r['difference'] for r in results['comparison']]

avg_mamdani = np.mean(mamdani_outputs)
avg_sugeno = np.mean(sugeno_outputs)
avg_difference = np.mean(differences)
max_difference = np.max(differences)
min_difference = np.min(differences)

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа 10: Сравнение методов Мамдани и Сугено</title>
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
        <p><strong>Выполнил:</strong> Тимошинов Егор Борисович</p>
        <p><strong>Группа:</strong> 16</p>
    </div>
    
    <h1>Лабораторная работа</h1>
    <h2>Сравнение методов Мамдани и Сугено</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение и сравнение двух основных методов нечеткого вывода: Мамдани и Сугено. Анализ особенностей каждого метода, их преимуществ и недостатков на примере задачи управления с двумя входными переменными.</p>
    
    <h3>Теоретические сведения</h3>
    
    <h4>Метод Мамдани</h4>
    <p>Метод Мамдани является одним из первых и наиболее распространенных методов нечеткого вывода. В этом методе как предпосылки, так и заключения правил представлены нечеткими множествами. Процесс вывода включает следующие этапы:</p>
    <ol>
        <li>Фаззификация входных переменных</li>
        <li>Вычисление степени истинности правил</li>
        <li>Агрегация выходных нечетких множеств</li>
        <li>Дефаззификация для получения четкого выходного значения</li>
    </ol>
    
    <p>Формула дефаззификации методом центра тяжести:</p>
    <div class="formula">
        $y = \\frac{{\\int x \\cdot \\mu(x) dx}}{{\\int \\mu(x) dx}}$
    </div>
    
    <h4>Метод Сугено</h4>
    <p>Метод Сугено (Такаги-Сугено) отличается от метода Мамдани тем, что выходные функции правил являются четкими функциями входных переменных, а не нечеткими множествами. Это упрощает процесс дефаззификации, так как не требуется этап дефаззификации - выход вычисляется как взвешенная сумма выходных функций правил.</p>
    
    <p>Формула вывода для системы Сугено первого порядка:</p>
    <div class="formula">
        $y = \\frac{{\\sum_{{i=1}}^{{n}} w_i \\cdot f_i(x_1, x_2, ..., x_m)}}{{\\sum_{{i=1}}^{{n}} w_i}}$
    </div>
    
    <p>где $w_i$ - степень истинности $i$-го правила, $f_i$ - выходная функция $i$-го правила.</p>
    
    <h3>Описание системы</h3>
    <p>Для сравнения методов была разработана нечеткая система управления с двумя входными переменными и одной выходной переменной:</p>
    <ul>
        <li><strong>Входная переменная 1:</strong> диапазон от 0 до 10, три терма: низкое, среднее, высокое</li>
        <li><strong>Входная переменная 2:</strong> диапазон от 0 до 10, три терма: низкое, среднее, высокое</li>
        <li><strong>Выходная переменная:</strong> диапазон от 0 до 20</li>
    </ul>
    
    <p>Система содержит 9 правил вывода, охватывающих все комбинации входных термов.</p>
    
    <h3>Функции принадлежности</h3>
    <p>Для входных переменных используются треугольные функции принадлежности:</p>
    {get_image('membership_plot', '<p>График функций принадлежности</p>')}
    <div class="figure-caption">Рисунок 1. Функции принадлежности входных переменных</div>
    
    <h2>Результаты сравнения</h2>
    
    <h3>Таблица результатов тестирования</h3>
    <table>
        <tr>
            <th>№</th>
            <th>Вход 1</th>
            <th>Вход 2</th>
            <th>Мамдани</th>
            <th>Сугено</th>
            <th>Разница</th>
        </tr>
        {comparison_table_rows}
    </table>
    
    <h3>Статистика сравнения</h3>
    <table>
        <tr>
            <th>Параметр</th>
            <th>Значение</th>
        </tr>
        <tr>
            <td>Среднее выходное значение (Мамдани)</td>
            <td>{avg_mamdani:.4f}</td>
        </tr>
        <tr>
            <td>Среднее выходное значение (Сугено)</td>
            <td>{avg_sugeno:.4f}</td>
        </tr>
        <tr>
            <td>Средняя разница между методами</td>
            <td>{avg_difference:.4f}</td>
        </tr>
        <tr>
            <td>Максимальная разница</td>
            <td>{max_difference:.4f}</td>
        </tr>
        <tr>
            <td>Минимальная разница</td>
            <td>{min_difference:.4f}</td>
        </tr>
    </table>
    
    <h3>Графическое сравнение выходных значений</h3>
    {get_image('comparison_plot', '<p>График сравнения</p>')}
    <div class="figure-caption">Рисунок 2. Сравнение выходных значений систем Мамдани и Сугено</div>
    
    <h3>График разницы между методами</h3>
    {get_image('difference_plot', '<p>График разницы</p>')}
    <div class="figure-caption">Рисунок 3. Абсолютная разница между выходными значениями методов</div>
    
    <h3>Поверхности выхода</h3>
    
    <h4>Система Мамдани</h4>
    {get_image('mamdani_surface', '<p>Поверхность Мамдани</p>')}
    <div class="figure-caption">Рисунок 4. Поверхность выхода системы Мамдани</div>
    
    <h4>Система Сугено</h4>
    {get_image('sugeno_surface', '<p>Поверхность Сугено</p>')}
    <div class="figure-caption">Рисунок 5. Поверхность выхода системы Сугено</div>
    
    <h2>Анализ результатов</h2>
    
    <h3>Преимущества метода Мамдани</h3>
    <ul>
        <li>Интуитивно понятная интерпретация правил</li>
        <li>Возможность визуализации выходных нечетких множеств</li>
        <li>Хорошо подходит для экспертных систем</li>
        <li>Позволяет анализировать степень неопределенности вывода</li>
    </ul>
    
    <h3>Преимущества метода Сугено</h3>
    <ul>
        <li>Более высокая вычислительная эффективность</li>
        <li>Отсутствие необходимости в дефаззификации</li>
        <li>Лучшая приспособляемость к данным (возможность обучения)</li>
        <li>Более плавные выходные поверхности</li>
        <li>Удобство использования в системах управления</li>
    </ul>
    
    <h3>Выводы</h3>
    <p>Проведенное сравнение показало, что оба метода дают сопоставимые результаты для рассматриваемой задачи. Метод Мамдани обеспечивает более интуитивную интерпретацию результатов, в то время как метод Сугено демонстрирует более высокую вычислительную эффективность и лучше подходит для задач, требующих адаптации параметров на основе данных.</p>
    
    <p>Выбор метода зависит от конкретной задачи: для экспертных систем с четкой интерпретацией правил предпочтительнее метод Мамдани, а для систем управления и задач оптимизации - метод Сугено.</p>
    
</body>
</html>"""

# Сохранение HTML файла
output_file = 'Timoshinov_E_B_Lab_10.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML отчет успешно создан: {output_file}")
