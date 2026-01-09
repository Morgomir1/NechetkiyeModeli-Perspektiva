import os
import pickle
import base64
import pandas as pd
import sys

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Загрузка результатов
print("Загрузка результатов...")
with open('sugeno_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
test_cases = data['test_cases']

# Проверка наличия изображений и их генерация, если нужно
required_images = ['temperature_membership.png', 'dT_membership.png', 
                  'sugeno_surface.png', 'test_results.png']
missing_images = [img for img in required_images if not os.path.exists(img)]

if missing_images:
    print(f"Обнаружены отсутствующие изображения: {', '.join(missing_images)}")
    print("Генерация графиков...")
    try:
        # Импортируем и запускаем генерацию графиков
        import generate_plots
        generate_plots.generate_all_plots()
        print("Графики успешно сгенерированы")
    except Exception as e:
        print(f"Ошибка при генерации графиков: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Преобразование изображений в base64
def image_to_base64(image_path):
    """Конвертирует изображение в base64 строку"""
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')
    return None

# Загрузка изображений
temp_membership_img = image_to_base64('temperature_membership.png')
dT_membership_img = image_to_base64('dT_membership.png')
sugeno_surface_img = image_to_base64('sugeno_surface.png')
test_results_img = image_to_base64('test_results.png')

# Создание таблицы результатов
results_table_rows = []
for i, result in enumerate(results, 1):
    results_table_rows.append(f"""
        <tr>
            <td>{i}</td>
            <td>{result['T']:.1f}</td>
            <td>{result['dT']:.1f}</td>
            <td>{result['P']:.2f}</td>
        </tr>""")

results_table = "\n".join(results_table_rows)

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа № 8: Проектирование нечетких систем Суджено</title>
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
    
    <h1>Лабораторная работа № 8</h1>
    <h2>Проектирование нечетких систем Суджено</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов проектирования нечетких систем вывода Суджено, построение функций принадлежности для входных переменных, определение правил вывода и анализ результатов работы системы.</p>
    
    <h3>Описание задачи</h3>
    <p>В данной работе рассматривается система управления температурой с использованием нечеткой логики Суджено. Система имеет две входные переменные:</p>
    <ul>
        <li><strong>Температура (T)</strong> - текущая температура в диапазоне от 0 до 50°C</li>
        <li><strong>Скорость изменения температуры (dT)</strong> - скорость изменения температуры в диапазоне от -5 до +5 град/мин</li>
    </ul>
    <p>Выходная переменная системы - <strong>Мощность нагревателя (P)</strong>, которая может принимать значения от 0 до 100%.</p>
    
    <h2>Описание системы</h2>
    
    <h3>Входные переменные и функции принадлежности</h3>
    
    <h4>Переменная "Температура"</h4>
    <p>Для переменной "Температура" определены три лингвистических терма:</p>
    <ul>
        <li><strong>Низкая</strong> - треугольная функция принадлежности с параметрами (0, 0, 20)</li>
        <li><strong>Средняя</strong> - треугольная функция принадлежности с параметрами (10, 25, 40)</li>
        <li><strong>Высокая</strong> - треугольная функция принадлежности с параметрами (30, 50, 50)</li>
    </ul>
    
    {f'<img src="data:image/png;base64,{temp_membership_img}" alt="Функции принадлежности для температуры">' if temp_membership_img else ''}
    <div class="figure-caption">Рисунок 1. Функции принадлежности для переменной "Температура"</div>
    
    <h4>Переменная "Скорость изменения температуры"</h4>
    <p>Для переменной "Скорость изменения температуры" определены три лингвистических терма:</p>
    <ul>
        <li><strong>Отрицательная</strong> - треугольная функция принадлежности с параметрами (-5, -5, 0)</li>
        <li><strong>Нулевая</strong> - треугольная функция принадлежности с параметрами (-2.5, 0, 2.5)</li>
        <li><strong>Положительная</strong> - треугольная функция принадлежности с параметрами (0, 5, 5)</li>
    </ul>
    
    {f'<img src="data:image/png;base64,{dT_membership_img}" alt="Функции принадлежности для скорости изменения температуры">' if dT_membership_img else ''}
    <div class="figure-caption">Рисунок 2. Функции принадлежности для переменной "Скорость изменения температуры"</div>
    
    <h3>Правила вывода Суджено</h3>
    <p>В системе Суджено выходные значения правил определяются как константы или функции входных переменных. В данной работе используются константные выходные значения.</p>
    
    <p>База правил системы состоит из 9 правил:</p>
    <table>
        <tr>
            <th>№</th>
            <th>Условие</th>
            <th>Выходное значение P, %</th>
        </tr>
        <tr>
            <td>1</td>
            <td>Если T низкая И dT отрицательная</td>
            <td>100</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Если T низкая И dT нулевая</td>
            <td>80</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Если T низкая И dT положительная</td>
            <td>60</td>
        </tr>
        <tr>
            <td>4</td>
            <td>Если T средняя И dT отрицательная</td>
            <td>70</td>
        </tr>
        <tr>
            <td>5</td>
            <td>Если T средняя И dT нулевая</td>
            <td>50</td>
        </tr>
        <tr>
            <td>6</td>
            <td>Если T средняя И dT положительная</td>
            <td>30</td>
        </tr>
        <tr>
            <td>7</td>
            <td>Если T высокая И dT отрицательная</td>
            <td>40</td>
        </tr>
        <tr>
            <td>8</td>
            <td>Если T высокая И dT нулевая</td>
            <td>20</td>
        </tr>
        <tr>
            <td>9</td>
            <td>Если T высокая И dT положительная</td>
            <td>0</td>
        </tr>
    </table>
    
    <h3>Метод вывода Суджено</h3>
    <p>В системе Суджено выходное значение вычисляется по формуле:</p>
    <div class="formula">
        $$P = \\frac{{\\sum_{{i=1}}^{{n}} w_i \\cdot z_i}}{{\\sum_{{i=1}}^{{n}} w_i}}$$
    </div>
    <p>где:</p>
    <ul>
        <li>$w_i$ - степень активации $i$-го правила (минимум значений функций принадлежности условий правила)</li>
        <li>$z_i$ - выходное значение $i$-го правила (константа или функция входных переменных)</li>
        <li>$n$ - количество правил</li>
    </ul>
    
    <h2>Результаты работы системы</h2>
    
    <h3>Поверхность вывода</h3>
    <p>Поверхность вывода показывает зависимость выходной переменной (мощность нагревателя) от входных переменных (температура и скорость изменения температуры).</p>
    
    {f'<img src="data:image/png;base64,{sugeno_surface_img}" alt="Поверхность вывода системы Суджено">' if sugeno_surface_img else ''}
    <div class="figure-caption">Рисунок 3. Поверхность вывода системы Суджено</div>
    
    <h3>Тестирование системы</h3>
    <p>Система была протестирована на нескольких тестовых примерах. Результаты представлены в таблице:</p>
    
    <table>
        <tr>
            <th>№</th>
            <th>Температура T, °C</th>
            <th>Скорость изменения dT, град/мин</th>
            <th>Мощность нагревателя P, %</th>
        </tr>
        {results_table}
    </table>
    
    {f'<img src="data:image/png;base64,{test_results_img}" alt="Результаты тестирования">' if test_results_img else ''}
    <div class="figure-caption">Рисунок 4. Результаты тестирования системы</div>
    
    <h2>Анализ результатов</h2>
    
    <p>Анализ результатов работы системы показывает следующее:</p>
    <ul>
        <li>При низкой температуре и отрицательной скорости изменения система выдает максимальную мощность (100%), что логично для быстрого нагрева.</li>
        <li>При высокой температуре и положительной скорости изменения система выдает минимальную мощность (0%), что предотвращает перегрев.</li>
        <li>Система плавно изменяет выходное значение в зависимости от входных переменных, что обеспечивает стабильное управление.</li>
        <li>Поверхность вывода имеет плавный характер без резких скачков, что является преимуществом системы Суджено.</li>
    </ul>
    
    <h2>Выводы</h2>
    
    <p>В результате выполнения лабораторной работы была спроектирована и реализована нечеткая система вывода Суджено для управления температурой. Система успешно работает с двумя входными переменными и выдает адекватные значения мощности нагревателя в зависимости от текущей температуры и скорости её изменения.</p>
    
    <p>Основные особенности системы Суджено:</p>
    <ul>
        <li>Выходные значения правил определяются как константы или функции входных переменных</li>
        <li>Вывод системы вычисляется как взвешенное среднее выходных значений правил</li>
        <li>Система обеспечивает плавное изменение выходного значения</li>
        <li>Поверхность вывода имеет непрерывный характер</li>
    </ul>
    
    <p>Система Суджено является эффективным инструментом для управления процессами, где требуется плавное изменение выходной переменной в зависимости от входных параметров.</p>
</body>
</html>
"""

# Сохранение HTML-отчета
output_file = 'Timoshinov_E_B_Lab_Sugeno.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML-отчет сохранен в файл: {output_file}")
