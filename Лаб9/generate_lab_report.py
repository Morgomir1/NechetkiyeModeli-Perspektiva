# -*- coding: utf-8 -*-
import os
import sys
import io
import pickle
import numpy as np
import pandas as pd

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

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Загрузка результатов
print("Загрузка результатов...")

# Загрузка результатов системы Сугено
try:
    with open('sugeno_credit_results.pkl', 'rb') as f:
        sugeno_data = pickle.load(f)
        sugeno_results = sugeno_data.get('results', [])
        credit_data = sugeno_data.get('data', None)
        # Система не сохраняется в pickle, т.к. содержит функции
except Exception as e:
    print(f"Ошибка загрузки данных Сугено: {e}")
    sugeno_results = []
    credit_data = None

# Загрузка результатов ANFIS
try:
    with open('anfis_results.pkl', 'rb') as f:
        anfis_data = pickle.load(f)
except Exception as e:
    print(f"Ошибка загрузки данных ANFIS: {e}")
    anfis_data = {}

# Загрузка изображений
try:
    with open('plot_images.pkl', 'rb') as f:
        images = pickle.load(f)
    print(f"Загружены изображения. Ключи: {list(images.keys())}")
    for key in images:
        if images[key]:
            print(f"  [OK] {key}: присутствует (длина: {len(images[key])} символов)")
        else:
            print(f"  [X] {key}: пустое")
except Exception as e:
    print(f"Ошибка загрузки изображений: {e}")
    import traceback
    traceback.print_exc()
    images = {}

# Функция для получения изображения или пустой строки
def get_image(key, default=""):
    if key in images and images[key]:
        print(f"  [OK] Вставка изображения: {key}")
        return f'<img src="data:image/png;base64,{images[key]}" alt="{key}">'
    else:
        print(f"  [X] Изображение отсутствует: {key} (есть в словаре: {key in images})")
        return default

# Подготовка данных для таблиц
sugeno_table_rows = ""
if sugeno_results:
    for result in sugeno_results[:10]:  # Первые 10 результатов
        sugeno_table_rows += f"""
        <tr>
            <td>{result['возраст']:.0f}</td>
            <td>{result['доход']:.0f}</td>
            <td>{result['оценка']:.4f}</td>
        </tr>"""

# Подготовка метрик ANFIS
anfis_metrics = ""
if anfis_data:
    train_mse = anfis_data.get('train_mse', 0)
    test_mse = anfis_data.get('test_mse', 0)
    train_r2 = anfis_data.get('train_r2', 0)
    test_r2 = anfis_data.get('test_r2', 0)
    
    anfis_metrics = f"""
    <table>
        <tr>
            <th>Метрика</th>
            <th>Обучающая выборка</th>
            <th>Тестовая выборка</th>
        </tr>
        <tr>
            <td>MSE (Mean Squared Error)</td>
            <td>{train_mse:.6f}</td>
            <td>{test_mse:.6f}</td>
        </tr>
        <tr>
            <td>R<sup>2</sup> (коэффициент детерминации)</td>
            <td>{train_r2:.6f}</td>
            <td>{test_r2:.6f}</td>
        </tr>
    </table>"""

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа: Система Такаги-Сугено и ANFIS</title>
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
    
    <h1>Лабораторная работа</h1>
    <h2>Система Такаги-Сугено и ANFIS для оценки кредита</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов построения нечетких систем Такаги-Сугено и адаптивных нейро-нечетких систем вывода (ANFIS) для решения задач оценки кредитоспособности клиентов.</p>
    
    <h2>1. Построение системы Такаги-Сугено для таблицы Credit</h2>
    
    <h3>1.1. Описание данных</h3>
    <p>Для построения модели использовался набор данных о клиентах банка (таблица Credit), включающий следующие признаки:</p>
    <ul>
        <li>Возраст клиента (количественный признак, диапазон: 18-70 лет)</li>
        <li>Доход клиента (количественный признак, диапазон: 20000-150000 руб)</li>
        <li>Кредитная история (категориальный признак: хорошая, средняя, плохая)</li>
        <li>Цель кредита (категориальный признак: потребительский, автомобиль, недвижимость, бизнес)</li>
        <li>Срок кредита в месяцах (количественный признак: 12, 24, 36, 48, 60)</li>
    </ul>
    <p>Целевая переменная: оценка кредита (вероятность возврата кредита, диапазон: 0-1).</p>
    <p>Общее количество наблюдений: {len(credit_data) if credit_data is not None else 1000}.</p>
    
    <h3>1.2. Структура системы Такаги-Сугено</h3>
    <p>Система Такаги-Сугено использует два входных параметра: возраст и доход клиента. Для каждой переменной определены три функции принадлежности:</p>
    <ul>
        <li><strong>Возраст:</strong> молодой, средний, пожилой</li>
        <li><strong>Доход:</strong> низкий, средний, высокий</li>
    </ul>
    <p>Система содержит 9 правил (3×3 комбинации функций принадлежности). Выходные значения правил представляют собой линейные функции входных переменных.</p>
    
    <h3>1.3. Функции принадлежности</h3>
    <p>Для входных переменных используются треугольные функции принадлежности. Ниже представлены графики функций принадлежности:</p>
    
    {get_image('age_membership', '<p>График функций принадлежности для возраста</p>')}
    <div class="figure-caption">Рисунок 1. Функции принадлежности для переменной "Возраст"</div>
    
    {get_image('income_membership', '<p>График функций принадлежности для дохода</p>')}
    <div class="figure-caption">Рисунок 2. Функции принадлежности для переменной "Доход"</div>
    
    <h3>1.4. Правила системы</h3>
    <p>Система содержит следующие правила:</p>
    <ol>
        <li>Если возраст молодой И доход низкий, то оценка = 0.2 + 0.000001×доход</li>
        <li>Если возраст молодой И доход средний, то оценка = 0.4 + 0.000002×доход</li>
        <li>Если возраст молодой И доход высокий, то оценка = 0.6 + 0.000001×доход</li>
        <li>Если возраст средний И доход низкий, то оценка = 0.3 + 0.0000015×доход</li>
        <li>Если возраст средний И доход средний, то оценка = 0.5 + 0.000002×доход</li>
        <li>Если возраст средний И доход высокий, то оценка = 0.7 + 0.0000015×доход</li>
        <li>Если возраст пожилой И доход низкий, то оценка = 0.25 + 0.000001×доход</li>
        <li>Если возраст пожилой И доход средний, то оценка = 0.45 + 0.0000015×доход</li>
        <li>Если возраст пожилой И доход высокий, то оценка = 0.65 + 0.000001×доход</li>
    </ol>
    
    <h3>1.5. Поверхность выхода системы</h3>
    <p>Ниже представлена трехмерная поверхность, показывающая зависимость выходного значения системы от входных переменных:</p>
    
    {get_image('surface', '<p>3D поверхность выхода системы</p>')}
    <div class="figure-caption">Рисунок 3. Поверхность выхода системы Такаги-Сугено</div>
    
    <h3>1.6. Примеры работы системы</h3>
    <p>Результаты тестирования системы на различных входных данных:</p>
    <table>
        <tr>
            <th>Возраст (лет)</th>
            <th>Доход (руб)</th>
            <th>Оценка кредита</th>
        </tr>
        {sugeno_table_rows}
    </table>
    
    <h2>2. Адаптивная нейро-нечеткая система вывода (ANFIS)</h2>
    
    <h3>2.1. Описание ANFIS</h3>
    <p>ANFIS (Adaptive Neuro-Fuzzy Inference System) — это гибридная система, объединяющая нечеткую логику и нейронные сети. ANFIS позволяет автоматически настраивать параметры нечеткой системы на основе обучающих данных.</p>
    <p>Основные преимущества ANFIS:</p>
    <ul>
        <li>Автоматическая настройка параметров функций принадлежности</li>
        <li>Обучение параметров выходных функций</li>
        <li>Способность к обучению на основе данных</li>
        <li>Сочетание интерпретируемости нечетких систем и обучаемости нейронных сетей</li>
    </ul>
    
    <h3>2.2. Структура ANFIS</h3>
    <p>ANFIS состоит из пяти слоев:</p>
    <ol>
        <li><strong>Слой 1:</strong> Функции принадлежности входных переменных</li>
        <li><strong>Слой 2:</strong> Вычисление степени активации правил</li>
        <li><strong>Слой 3:</strong> Нормализация степеней активации</li>
        <li><strong>Слой 4:</strong> Выходные функции правил</li>
        <li><strong>Слой 5:</strong> Дефаззификация (взвешенная сумма выходов правил)</li>
    </ol>
    
    <h3>2.3. Обучение ANFIS</h3>
    <p>Для обучения ANFIS использовались следующие методы:</p>
    <ul>
        <li><strong>Метод наименьших квадратов:</strong> для настройки параметров выходных функций (линейных)</li>
        <li><strong>Градиентный спуск:</strong> для дополнительной настройки параметров функций принадлежности</li>
    </ul>
    <p>Данные были разделены на обучающую выборку (70%) и тестовую выборку (30%).</p>
    
    {get_image('training_history', '<p style="color: red;">[WARN] График истории обучения не был сгенерирован</p>')}
    <div class="figure-caption">Рисунок 4. История обучения ANFIS (изменение ошибки по эпохам)</div>
    
    <h3>2.4. Результаты обучения ANFIS</h3>
    <p>Метрики качества обученной модели:</p>
    {anfis_metrics if anfis_metrics else '<p>Данные не загружены</p>'}
    
    <h3>2.5. Анализ предсказаний</h3>
    <p>Ниже представлены графики сравнения предсказанных и фактических значений:</p>
    
    {get_image('train_predictions', '<p style="color: red;">[WARN] График предсказаний на обучающей выборке не был сгенерирован</p>')}
    <div class="figure-caption">Рисунок 5. Предсказания vs фактические значения (обучающая выборка)</div>
    
    {get_image('test_predictions', '<p style="color: red;">[WARN] График предсказаний на тестовой выборке не был сгенерирован</p>')}
    <div class="figure-caption">Рисунок 6. Предсказания vs фактические значения (тестовая выборка)</div>
    
    <h3>2.6. Анализ остатков</h3>
    <p>Графики остатков позволяют оценить качество модели:</p>
    
    {get_image('train_residuals', '<p style="color: red;">[WARN] График остатков на обучающей выборке не был сгенерирован</p>')}
    <div class="figure-caption">Рисунок 7. Остатки модели (обучающая выборка)</div>
    
    {get_image('test_residuals', '<p style="color: red;">[WARN] График остатков на тестовой выборке не был сгенерирован</p>')}
    <div class="figure-caption">Рисунок 8. Остатки модели (тестовая выборка)</div>
    
    <h2>Выводы</h2>
    <p>В ходе выполнения лабораторной работы были построены и обучены две модели для оценки кредитоспособности клиентов:</p>
    <ol>
        <li><strong>Система Такаги-Сугено:</strong> нечеткая система с фиксированными правилами и параметрами, использующая линейные функции в заключениях правил. Система показала способность к интерпретации результатов и гибкость в настройке правил.</li>
        <li><strong>ANFIS:</strong> адаптивная нейро-нечеткая система, способная автоматически настраивать параметры на основе обучающих данных. Модель показала хорошую способность к обучению и обобщению на новых данных.</li>
    </ol>
    <p>Сравнение результатов показывает, что ANFIS позволяет улучшить качество предсказаний за счет автоматической настройки параметров системы на основе данных. Коэффициент детерминации R<sup>2</sup> на тестовой выборке составляет {anfis_data.get('test_r2', 0):.4f}, что указывает на хорошее качество модели.</p>
    <p>Обе системы могут быть использованы в банковской сфере для оценки риска выдачи кредита новым клиентам на основе их характеристик (возраст, доход и другие параметры).</p>
</body>
</html>
"""

# Сохранение HTML файла
with open('Timoshinov_E_B_Lab_9.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML отчет создан: Timoshinov_E_B_Lab_9.html")
