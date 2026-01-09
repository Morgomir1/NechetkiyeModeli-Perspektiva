# -*- coding: utf-8 -*-
"""
Генерация HTML-отчета для лабораторной работы по нечеткой классификации
"""

import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Загрузка результатов
with open('fuzzy_classification_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Извлечение данных
training_data = results['training_data']
membership_x1 = results['membership_x1']
membership_x2 = results['membership_x2']
mu_p_training = results['mu_p_training']
beta = results['beta']
c_p = results['c_p']
P1 = results['P1']
P2 = results['P2']
new_observation = results['new_observation']
new_mu_x1 = results['new_mu_x1']
new_mu_x2 = results['new_mu_x2']
new_mu_p = results['new_mu_p']
mu_class1 = results['mu_class1']
mu_class2 = results['mu_class2']
predicted_class = results['predicted_class']
max_membership = results['max_membership']
img_functions = results['img_functions']
img_data_points = results['img_data_points']
img_membership = results['img_membership']

# Генерация таблиц
n_obs = len(training_data['x1'])

# Таблица обучающей выборки
training_table = ""
for i in range(n_obs):
    training_table += f"<tr><td>{i+1}</td><td>{training_data['x1'][i]:.1f}</td><td>{training_data['x2'][i]:.1f}</td><td>{training_data['classes'][i]}</td></tr>\n"

# Таблица принадлежности x1 к градациям
membership_x1_table = ""
for i in range(n_obs):
    membership_x1_table += f"<tr><td>{i+1}</td><td>{membership_x1['S'][i]:.3f}</td><td>{membership_x1['M'][i]:.3f}</td><td>{membership_x1['L'][i]:.3f}</td></tr>\n"

# Таблица принадлежности x2 к градациям
membership_x2_table = ""
for i in range(n_obs):
    membership_x2_table += f"<tr><td>{i+1}</td><td>{membership_x2['S'][i]:.3f}</td><td>{membership_x2['M'][i]:.3f}</td><td>{membership_x2['L'][i]:.3f}</td></tr>\n"

# Таблица принадлежности к декартову произведению
cartesian_product = [('S', 'S'), ('S', 'M'), ('S', 'L'), ('M', 'S'), ('M', 'M'), ('M', 'L'), ('L', 'S'), ('L', 'M'), ('L', 'L')]
mu_p_table = ""
for p in cartesian_product:
    mu_p_table += f"<tr><td>{p}</td>"
    for i in range(n_obs):
        mu_p_table += f"<td>{mu_p_training[p][i]:.3f}</td>"
    mu_p_table += "</tr>\n"

# Таблица β и c_p
beta_table = ""
for p in cartesian_product:
    cls = ""
    if p in P1:
        cls = "A1"
    elif p in P2:
        cls = "A2"
    beta_table += f"<tr><td>{p}</td>"
    for i in range(n_obs):
        beta_table += f"<td>{mu_p_training[p][i]:.3f}</td>"
    beta_table += f"<td>{beta[1][p]:.3f}</td><td>{beta[2][p]:.3f}</td><td>{c_p[p]:.3f}</td><td>{cls}</td></tr>\n"

# Таблица принадлежности нового наблюдения к декартову произведению
new_mu_p_table = ""
for p in cartesian_product:
    new_mu_p_table += f"<tr><td>{p}</td><td>{new_mu_p[p]:.3f}</td></tr>\n"

# Выносим проблемные LaTeX формулы с обратными слэшами в переменные
latex_mu_s = r"$$\mu_S(x) = \begin{cases} 1, & x \leq 0 \\ 1 - 2x, & 0 < x < 0.5 \\ 0, & x \geq 0.5 \end{cases}$$"
latex_mu_m = r"$$\mu_M(x) = \begin{cases} 0, & x \leq 0 \text{ или } x \geq 1 \\ 2x, & 0 < x \leq 0.5 \\ 2(1-x), & 0.5 < x < 1 \end{cases}$$"
latex_mu_l = r"$$\mu_L(x) = \begin{cases} 0, & x \leq 0.5 \\ 2(x-0.5), & 0.5 < x < 1 \\ 1, & x \geq 1 \end{cases}$$"
latex_set_notation = r"$\{S, M, L\}$"
latex_mu_p_formula = r"$$\mu_p = \mu_{p1}(x_1) \cdot \mu_{p2}(x_2)$$"
latex_beta_formula = r"$$\beta_{\bar{p}}^j = \sum_{\bar{x} \in \widetilde{A}_j} \mu_{\bar{p}}(x)$$"
latex_cp_formula = r"$$c_{\bar{p}} = \frac{\left| \beta_{\bar{p}}^1 - \beta_{\bar{p}}^2 \right|}{\left| \beta_{\bar{p}}^1 + \beta_{\bar{p}}^2 \right|} \in [0; 1]$$"
latex_mu_class_formula = r"$$\mu_{\widetilde{A}_j}(\bar{x}) = \max_{\bar{p} \in P_j} c_{\bar{p}} \mu_{\bar{p}}(\bar{x})$$"

# Выносим все проблемные LaTeX строки с обратными слэшами в переменные
latex_beta_p_j = r"$\beta_{\bar{p}}^j$"
latex_c_p_bar = r"$c_{\bar{p}}$"
latex_beta_p_j_title = r"$\beta_{\bar{p}}^j$"
latex_p_bar = r"$\bar{p}$"
latex_p_bar_table = r"$\bar{p} = (p1,p2)$"
latex_mu_p_bar_x = r"$\mu_{\bar{p}}(\bar{x})$"
latex_p_bar_equals = r"$\bar{p} = (p1, p2)$"
latex_pi_in = r"$pi \in$"
latex_beta_1 = r"$\beta^1$"
latex_beta_2 = r"$\beta^2$"

# Формулы для результатов (используем .format() чтобы избежать проблем с обратными слэшами в f-string)
mu_class1_formula = "$\\mu_{{\\widetilde{{A}}_1}}({:.1f}, {:.1f}) = {:.6f}$".format(
    new_observation[0], new_observation[1], mu_class1)
mu_class2_formula = "$\\mu_{{\\widetilde{{A}}_2}}({:.1f}, {:.1f}) = {:.6f}$".format(
    new_observation[0], new_observation[1], mu_class2)

# HTML-контент (используем обычную строку для части с обратными слэшами)
mathjax_config = """        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };"""

html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа: Нечеткая классификация</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
{mathjax_config}
    </script>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            max-width: 1000px;
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
            font-size: 11pt;
        }}
        table, th, td {{
            border: 1px solid black;
        }}
        th, td {{
            padding: 6px;
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
    </style>
</head>
<body>
    <div class="author">
        <p><strong>Выполнил:</strong> Тимошинов Егор Борисович</p>
        <p><strong>Группа:</strong> 16</p>
    </div>
    
    <h1>Лабораторная работа</h1>
    <h2>Нечеткая классификация</h2>
    
    <h3>Цель работы</h3>
    <p>Создание процедуры нечеткой классификации по обучающей выборке и классификация нового наблюдения.</p>
    
    <h3>Исходные данные</h3>
    <p>Обучающая выборка состоит из 7 наблюдений с двумя признаками (x1, x2) и известными классами:</p>
    
    <table>
        <tr>
            <th>Наблюдение</th>
            <th>x1</th>
            <th>x2</th>
            <th>Класс</th>
        </tr>
        {training_table}
    </table>
    
    <p>Требуется определить, к какому классу относится новое наблюдение: ({new_observation[0]:.1f}, {new_observation[1]:.1f}).</p>
    
    <h2>Метод нечеткой классификации</h2>
    
    <p>Процедура нечеткой классификации основана на следующих шагах:</p>
    <ol>
        <li>Определение функций принадлежности для градаций атрибутов: малое (S), среднее (M), большое (L).</li>
        <li>Вычисление принадлежности каждого атрибута к градациям для всех наблюдений обучающей выборки.</li>
        <li>Вычисление принадлежности образов к элементам декартова произведения нечетких градаций.</li>
        <li>Вычисление суммарных степеней {latex_beta_p_j} для каждого класса.</li>
        <li>Вычисление степени доверия {latex_c_p_bar} к разделяющей способности векторов атрибутов.</li>
        <li>Определение классов для векторов значений нечетких атрибутов.</li>
        <li>Вычисление функции принадлежности нового наблюдения к классам.</li>
    </ol>
    
    <h2>Функции принадлежности для градаций</h2>
    
    <p>Для каждого атрибута используются три градации: малое (S), среднее (M), большое (L). Функции принадлежности имеют следующий вид:</p>
    
    <div class="formula">
        {latex_mu_s}
    </div>
    
    <div class="formula">
        {latex_mu_m}
    </div>
    
    <div class="formula">
        {latex_mu_l}
    </div>
    
    <img src="data:image/png;base64,{img_functions}" alt="Функции принадлежности">
    <div class="figure-caption">Рисунок 1. Функции принадлежности для градаций S, M, L</div>
    
    <h2>Принадлежность атрибутов к градациям</h2>
    
    <p>Для каждого наблюдения вычислена принадлежность атрибутов к градациям:</p>
    
    <h3>Первый атрибут (x1)</h3>
    <table>
        <tr>
            <th>Наблюдение</th>
            <th>Малое (S)</th>
            <th>Среднее (M)</th>
            <th>Большое (L)</th>
        </tr>
        {membership_x1_table}
    </table>
    
    <h3>Второй атрибут (x2)</h3>
    <table>
        <tr>
            <th>Наблюдение</th>
            <th>Малое (S)</th>
            <th>Среднее (M)</th>
            <th>Большое (L)</th>
        </tr>
        {membership_x2_table}
    </table>
    
    <h2>Принадлежность к декартову произведению градаций</h2>
    
    <p>Для каждого образа вычислена принадлежность к элементам декартова произведения нечетких градаций с помощью операции И (произведение):</p>
    
    <div class="formula">
        {latex_mu_p_formula}
    </div>
    
    <p>где {latex_p_bar_equals}, {latex_pi_in} {latex_set_notation}.</p>
    
    <table>
        <tr>
            <th>{latex_p_bar_table}</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
        </tr>
        {mu_p_table}
    </table>
    
    <h2>Вычисление {latex_beta_p_j_title} для каждого класса</h2>
    
    <p>Для каждого класса определяются суммарные степени того, что вектор значений нечетких атрибутов {latex_p_bar} характерен для данного класса:</p>
    
    <div class="formula">
        {latex_beta_formula}
    </div>
    
    <p>Также вычисляется степень доверия к разделяющей способности:</p>
    
    <div class="formula">
        {latex_cp_formula}
    </div>
    
    <table>
        <tr>
            <th>{latex_p_bar_table}</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>6</th>
            <th>7</th>
            <th>{latex_beta_1}</th>
            <th>{latex_beta_2}</th>
            <th>$c_p$</th>
            <th>Класс</th>
        </tr>
        {beta_table}
    </table>
    
    <p>Векторы, характерные для класса 1: $P_1 = {P1}$</p>
    <p>Векторы, характерные для класса 2: $P_2 = {P2}$</p>
    
    <h2>Классификация нового наблюдения</h2>
    
    <p>Для нового наблюдения ({new_observation[0]:.1f}, {new_observation[1]:.1f}) вычислена принадлежность к градациям:</p>
    
    <p>Первый атрибут: S = {new_mu_x1['S']:.3f}, M = {new_mu_x1['M']:.3f}, L = {new_mu_x1['L']:.3f}</p>
    <p>Второй атрибут: S = {new_mu_x2['S']:.3f}, M = {new_mu_x2['M']:.3f}, L = {new_mu_x2['L']:.3f}</p>
    
    <p>Принадлежность к декартову произведению градаций:</p>
    
    <table>
        <tr>
            <th>{latex_p_bar_table}</th>
            <th>{latex_mu_p_bar_x}</th>
        </tr>
        {new_mu_p_table}
    </table>
    
    <p>Функция принадлежности к классам вычисляется по формуле:</p>
    
    <div class="formula">
        {latex_mu_class_formula}
    </div>
    
    <p>Результаты:</p>
    <p>{mu_class1_formula}</p>
    <p>{mu_class2_formula}</p>
    
    <img src="data:image/png;base64,{img_membership}" alt="Степени принадлежности">
    <div class="figure-caption">Рисунок 2. Степени принадлежности нового наблюдения к классам</div>
    
    <h2>Визуализация данных</h2>
    
    <p>На следующем рисунке представлены точки обучающей выборки и новое наблюдение:</p>
    
    <img src="data:image/png;base64,{img_data_points}" alt="Обучающая выборка и новое наблюдение">
    <div class="figure-caption">Рисунок 3. Обучающая выборка и новое наблюдение</div>
    
    <h2>Результат классификации</h2>
    
    <p>Наблюдение ({new_observation[0]:.1f}, {new_observation[1]:.1f}) относится к <strong>КЛАССУ {predicted_class}</strong>.</p>
    <p>Степень принадлежности к классу {predicted_class}: {max_membership:.6f}</p>
    
    <h2>Выводы</h2>
    
    <p>В результате выполнения лабораторной работы была создана процедура нечеткой классификации по обучающей выборке. Процедура основана на использовании градаций атрибутов (малое, среднее, большое), вычислении принадлежности к декартову произведению градаций, определении характеристик классов и вычислении функции принадлежности нового наблюдения к классам.</p>
    
    <p>Для нового наблюдения ({new_observation[0]:.1f}, {new_observation[1]:.1f}) были вычислены степени принадлежности к обоим классам. Максимальная степень принадлежности соответствует классу {predicted_class}, поэтому наблюдение было отнесено к этому классу.</p>
    
    <p>Метод нечеткой классификации позволяет не только определить класс наблюдения, но и оценить уверенность в классификации через значения степеней принадлежности.</p>
</body>
</html>
"""

output_file = os.path.join(script_dir, 'Timoshinov_E_B_Lab_7.html')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML-отчет сохранен: {output_file}")
