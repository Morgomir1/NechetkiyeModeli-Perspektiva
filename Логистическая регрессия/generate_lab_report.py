import os
import sys
import pickle
import pandas as pd
import base64
from sklearn.metrics import classification_report

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Загрузка результатов
print("Загрузка результатов...")
with open('logistic_regression_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Загрузка данных
data = pd.read_csv('logistic_regression_data.csv', encoding='utf-8-sig')

# Преобразование изображений в base64
def image_to_base64(image_path):
    """Конвертирует изображение в base64 строку"""
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')
    return None

# Загрузка изображений
logistic_curve_img = image_to_base64('logistic_curve.png')
roc_curve_img = image_to_base64('roc_curve.png')
confusion_matrix_img = image_to_base64('confusion_matrix.png')
probability_distribution_img = image_to_base64('probability_distribution.png')

# Создание таблицы с данными (первые 20 строк)
data_table_rows = []
for idx, row in data.head(20).iterrows():
    data_table_rows.append(f"""
        <tr>
            <td>{int(row['age'])}</td>
            <td>{int(row['disease'])}</td>
            <td>{row['predicted_probability']:.4f}</td>
            <td>{int(row['predicted_class'])}</td>
        </tr>""")

data_table = "\n".join(data_table_rows)

# Классификационный отчет
y_true = data['disease'].values
y_pred = data['predicted_class'].values
class_report = classification_report(y_true, y_pred, output_dict=True)

# Подготовка всех значений для подстановки
n_samples = len(data)
n_disease = data['disease'].sum()
n_no_disease = (data['disease'] == 0).sum()
avg_age = data['age'].mean()
min_age = data['age'].min()
max_age = data['age'].max()
intercept = results['coefficients']['intercept']
coefficient = results['coefficients']['coefficient']
roc_auc = results['roc_auc']
tn = results['confusion_matrix'][0][0]
fp = results['confusion_matrix'][0][1]
fn = results['confusion_matrix'][1][0]
tp = results['confusion_matrix'][1][1]
accuracy = class_report['accuracy']
precision = class_report['1']['precision']
recall = class_report['1']['recall']
f1_score = class_report['1']['f1-score']

# Подготовка LaTeX формул как переменных (чтобы избежать проблем с обратными слэшами в f-string)
latex_formula1 = r"$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$"
latex_beta0 = r"$\beta_0$"
latex_beta1 = r"$\beta_1$"
# Формула с подстановкой значений
latex_text_disease = r"\text{заболевание}"
latex_text_age = r"\text{возраст}"
latex_formula2 = f"$P({latex_text_disease}|{latex_text_age}) = \\frac{{1}}{{1 + e^{{-({intercept:.4f} + {coefficient:.4f} \\cdot {latex_text_age})}}}}$"

# Формирование HTML
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа: Логистическая регрессия</title>
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
    <h2>Логистическая регрессия</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов построения моделей логистической регрессии и оценки их качества с помощью ROC-анализа.</p>
    
    <h3>Теоретические сведения</h3>
    <p>Логистическая регрессия используется для моделирования зависимости бинарной зависимой переменной от независимых переменных. В отличие от линейной регрессии, логистическая регрессия предсказывает вероятность того, что зависимая переменная примет значение 1.</p>
    
    <p>Модель логистической регрессии имеет вид:</p>
    <div class="formula">
        {latex_formula1}
    </div>
    
    <p>где $P(Y=1|X)$ — вероятность того, что зависимая переменная $Y$ равна 1 при заданных значениях независимых переменных $X$, {latex_beta0} — свободный член, {latex_beta1} — коэффициент регрессии.</p>
    
    <h3>Задание</h3>
    <ol>
        <li>Построить модель логистической регрессии для предсказания вероятности заболевания в зависимости от возраста.</li>
        <li>Построить график зависимости вероятности заболевания от возраста.</li>
        <li>Построить ROC-кривую и вычислить значение AUC.</li>
        <li>Построить матрицу ошибок и проанализировать качество классификации.</li>
        <li>Проанализировать распределение предсказанных вероятностей.</li>
    </ol>
    
    <h2>Результаты выполнения задания</h2>
    
    <h3>Исходные данные</h3>
    <p>Для анализа использованы данные о {n_samples} наблюдениях. Зависимая переменная — наличие заболевания (0 — нет заболевания, 1 — есть заболевание), независимая переменная — возраст.</p>
    
    <p>Статистика по данным:</p>
    <ul>
        <li>Количество случаев заболевания: {n_disease}</li>
        <li>Количество случаев без заболевания: {n_no_disease}</li>
        <li>Средний возраст: {avg_age:.2f} лет</li>
        <li>Минимальный возраст: {min_age} лет</li>
        <li>Максимальный возраст: {max_age} лет</li>
    </ul>
    
    <p>Первые 20 наблюдений:</p>
    <table>
        <tr>
            <th>Возраст</th>
            <th>Заболевание</th>
            <th>Предсказанная вероятность</th>
            <th>Предсказанный класс</th>
        </tr>
        {data_table}
    </table>
    
    <h3>Результаты моделирования</h3>
    <p>Коэффициенты модели логистической регрессии:</p>
    <ul>
        <li>Свободный член ({latex_beta0}): {intercept:.4f}</li>
        <li>Коэффициент при возрасте ({latex_beta1}): {coefficient:.4f}</li>
    </ul>
    
    <p>Уравнение модели:</p>
    <div class="formula">
        {latex_formula2}
    </div>
    
    <div class="task">
        <h3>График зависимости вероятности заболевания от возраста</h3>
        <img src="data:image/png;base64,{logistic_curve_img}" alt="Логистическая кривая">
        <div class="figure-caption">Рисунок 1. Зависимость вероятности заболевания от возраста</div>
        <p>На графике видно, что вероятность заболевания увеличивается с возрастом. Логистическая кривая имеет S-образную форму, что характерно для логистической регрессии.</p>
    </div>
    
    <div class="task">
        <h3>ROC-кривая</h3>
        <img src="data:image/png;base64,{roc_curve_img}" alt="ROC-кривая">
        <div class="figure-caption">Рисунок 2. ROC-кривая</div>
        <p>Площадь под ROC-кривой (AUC) составляет {roc_auc:.4f}. Значение AUC близкое к 1 указывает на хорошее качество модели. Значение 0.5 соответствует случайному классификатору.</p>
    </div>
    
    <div class="task">
        <h3>Матрица ошибок</h3>
        <img src="data:image/png;base64,{confusion_matrix_img}" alt="Матрица ошибок">
        <div class="figure-caption">Рисунок 3. Матрица ошибок</div>
        <p>Матрица ошибок показывает количество правильных и неправильных классификаций:</p>
        <ul>
            <li>True Negative (TN): {tn} — правильно предсказано отсутствие заболевания</li>
            <li>False Positive (FP): {fp} — ошибочно предсказано наличие заболевания</li>
            <li>False Negative (FN): {fn} — ошибочно предсказано отсутствие заболевания</li>
            <li>True Positive (TP): {tp} — правильно предсказано наличие заболевания</li>
        </ul>
    </div>
    
    <div class="task">
        <h3>Распределение предсказанных вероятностей</h3>
        <img src="data:image/png;base64,{probability_distribution_img}" alt="Распределение вероятностей">
        <div class="figure-caption">Рисунок 4. Распределение предсказанных вероятностей</div>
        <p>Гистограмма показывает распределение предсказанных вероятностей для случаев с заболеванием и без него. Хорошая модель должна давать низкие вероятности для случаев без заболевания и высокие — для случаев с заболеванием.</p>
    </div>
    
    <h3>Оценка качества модели</h3>
    <p>Метрики качества классификации:</p>
    <ul>
        <li>Точность (Accuracy): {accuracy:.4f}</li>
        <li>Precision (для класса 1): {precision:.4f}</li>
        <li>Recall (для класса 1): {recall:.4f}</li>
        <li>F1-score (для класса 1): {f1_score:.4f}</li>
    </ul>
    
    <h3>Выводы</h3>
    <p>В ходе выполнения лабораторной работы была построена модель логистической регрессии для предсказания вероятности заболевания в зависимости от возраста. Модель показала хорошее качество (AUC = {roc_auc:.4f}), что указывает на наличие значимой связи между возрастом и вероятностью заболевания. ROC-кривая демонстрирует, что модель лучше случайного классификатора. Матрица ошибок и распределение вероятностей подтверждают адекватность построенной модели.</p>
    
</body>
</html>"""

# Сохранение HTML файла
with open('Timoshinov_E_B_Lab_LogisticRegression.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML отчет успешно создан: Timoshinov_E_B_Lab_LogisticRegression.html")

