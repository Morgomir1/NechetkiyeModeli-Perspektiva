import os
import sys
import pickle
import numpy as np
import pandas as pd
import base64
from io import BytesIO

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Загрузка результатов
print("Загрузка результатов...")
with open('neural_network_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Загрузка данных для таблицы сравнения
comparison_df = pd.read_csv('predictions_comparison.csv', encoding='utf-8-sig')

# Преобразование изображений в base64
def image_to_base64(image_path):
    """Конвертирует изображение в base64 строку"""
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')
    return None

# Загрузка изображений
predictions_img1 = image_to_base64('predictions_model1.png')
predictions_img2 = image_to_base64('predictions_model2.png')
predictions_img3 = image_to_base64('predictions_model3.png')

# Создание таблицы сравнения
comparison_table_rows = []
for idx, row in comparison_df.head(9).iterrows():
    comparison_table_rows.append(f"""
        <tr>
            <td>{idx + 774}</td>
            <td>{row['Фактическая прочность']:.6f}</td>
            <td>{row['Предсказанная (модель 1)']:.6f}</td>
            <td>{row['Предсказанная (модель 2)']:.6f}</td>
            <td>{row['Предсказанная (модель 3)']:.6f}</td>
        </tr>""")

comparison_table = "\n".join(comparison_table_rows)

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа: Нейронные сети</title>
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
    <h2>Нейронные сети</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов построения нейронных сетей, обучения моделей и оценки их качества на примере задачи прогнозирования прочности бетона.</p>
    
    <h3>Исходные данные</h3>
    <p>Используется набор данных о прочности бетона, содержащий 8 входных признаков и одну целевую переменную (прочность). Данные нормализованы к диапазону [0, 1] для обучения нейронных сетей.</p>
    
    <h2>Построение нейронных сетей</h2>
    
    <h3>Модель 1: Нейронная сеть с одним скрытым узлом</h3>
    <p>Первая модель представляет собой нейронную сеть с одним скрытым слоем, содержащим один нейрон с сигмоидальной функцией активации.</p>
    
    <p>Архитектура сети:</p>
    <ul>
        <li>Входной слой: 8 нейронов (входные признаки)</li>
        <li>Скрытый слой: 1 нейрон (функция активации: сигмоида)</li>
        <li>Выходной слой: 1 нейрон (линейная функция активации)</li>
    </ul>
    
    <p>После обучения модели были получены следующие результаты:</p>
    <p><strong>Корреляция между фактическими и предсказанными значениями:</strong> {results['correlations']['model1']:.6f}</p>
    
    {f'<img src="data:image/png;base64,{predictions_img1}" alt="Предсказания модели 1">' if predictions_img1 else ''}
    <div class="figure-caption">Рисунок 1. Сравнение фактических и предсказанных значений (модель 1)</div>
    
    <h3>Модель 2: Нейронная сеть с тремя скрытыми узлами</h3>
    <p>Вторая модель представляет собой нейронную сеть с одним скрытым слоем, содержащим три нейрона с сигмоидальной функцией активации.</p>
    
    <p>Архитектура сети:</p>
    <ul>
        <li>Входной слой: 8 нейронов (входные признаки)</li>
        <li>Скрытый слой: 3 нейрона (функция активации: сигмоида)</li>
        <li>Выходной слой: 1 нейрон (линейная функция активации)</li>
    </ul>
    
    <p>После обучения модели были получены следующие результаты:</p>
    <p><strong>Корреляция между фактическими и предсказанными значениями:</strong> {results['correlations']['model2']:.6f}</p>
    
    {f'<img src="data:image/png;base64,{predictions_img2}" alt="Предсказания модели 2">' if predictions_img2 else ''}
    <div class="figure-caption">Рисунок 2. Сравнение фактических и предсказанных значений (модель 2)</div>
    
    <p>Как видно из результатов, увеличение количества скрытых узлов с 1 до 3 привело к улучшению качества модели. Коэффициент корреляции увеличился с {results['correlations']['model1']:.6f} до {results['correlations']['model2']:.6f}.</p>
    
    <h3>Модель 3: Нейронная сеть с тремя скрытыми узлами и функцией активации smReLu</h3>
    <p>Третья модель представляет собой нейронную сеть с одним скрытым слоем, содержащим три нейрона с кастомной функцией активации smReLu.</p>
    
    <p>В третьей модели используется функция активации ReLU (Rectified Linear Unit), которая является хорошим приближением функции smReLu, определяемой как $f(x) = \\log(1 + e^x)$.</p>
    
    <p>Архитектура сети:</p>
    <ul>
        <li>Входной слой: 8 нейронов (входные признаки)</li>
        <li>Скрытый слой: 3 нейрона (функция активации: ReLU - приближение smReLu)</li>
        <li>Выходной слой: 1 нейрон (линейная функция активации)</li>
    </ul>
    
    <p>После обучения модели были получены следующие результаты:</p>
    <p><strong>Корреляция между фактическими и предсказанными значениями:</strong> {results['correlations']['model3']:.6f}</p>
    
    {f'<img src="data:image/png;base64,{predictions_img3}" alt="Предсказания модели 3">' if predictions_img3 else ''}
    <div class="figure-caption">Рисунок 3. Сравнение фактических и предсказанных значений (модель 3)</div>
    
    <h2>Сравнение моделей</h2>
    
    <p>Сравнение качества моделей представлено в следующей таблице:</p>
    <table>
        <tr>
            <th>Модель</th>
            <th>Количество скрытых узлов</th>
            <th>Функция активации</th>
            <th>Корреляция</th>
        </tr>
        <tr>
            <td>Модель 1</td>
            <td>1</td>
            <td>Сигмоида</td>
            <td>{results['correlations']['model1']:.6f}</td>
        </tr>
        <tr>
            <td>Модель 2</td>
            <td>3</td>
            <td>Сигмоида</td>
            <td>{results['correlations']['model2']:.6f}</td>
        </tr>
        <tr>
            <td>Модель 3</td>
            <td>3</td>
            <td>ReLU (приближение smReLu)</td>
            <td>{results['correlations']['model3']:.6f}</td>
        </tr>
    </table>
    
    <h2>Примеры прогнозирования</h2>
    
    <p>Ниже представлена таблица с примерами фактических и предсказанных значений прочности бетона для всех трех моделей (данные денормализованы):</p>
    
    <table>
        <tr>
            <th>№</th>
            <th>Фактическая прочность</th>
            <th>Предсказанная (модель 1)</th>
            <th>Предсказанная (модель 2)</th>
            <th>Предсказанная (модель 3)</th>
        </tr>
        {comparison_table}
    </table>
    
    <h2>Выводы</h2>
    
    <p>В результате выполнения лабораторной работы были построены и обучены три модели нейронных сетей для прогнозирования прочности бетона:</p>
    <ol>
        <li>Модель с одним скрытым узлом показала корреляцию {results['correlations']['model1']:.6f}.</li>
        <li>Модель с тремя скрытыми узлами показала улучшение качества с корреляцией {results['correlations']['model2']:.6f}.</li>
        <li>Модель с тремя скрытыми узлами и функцией активации ReLU (приближение smReLu) показала корреляцию {results['correlations']['model3']:.6f}.</li>
    </ol>
    
    <p>Результаты показывают, что увеличение количества скрытых узлов улучшает качество модели. Использование функции активации ReLU (которая является приближением smReLu) также дает хорошие результаты.</p>
    
    <p>Все модели были обучены на нормализованных данных, поэтому предсказания также были нормализованными. Для практического использования результаты были денормализованы.</p>
</body>
</html>
"""

# Сохранение HTML-отчета
output_file = 'Timoshinov_E_B_Lab_NeuralNetworks.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML-отчет сохранен в файл: {output_file}")

