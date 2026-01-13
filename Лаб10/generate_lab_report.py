# -*- coding: utf-8 -*-
"""
Генерация HTML отчета для лабораторной работы 10
"""

import numpy as np
import pickle
import os

def format_matrix_html(M, states, title="M"):
    """Форматирует матрицу в HTML таблицу"""
    html = f'<p><strong>{title}:</strong></p>\n'
    html += '<div class="formula">\n'
    html += f'    {title} = <table style="width: auto; margin: 10px auto;">\n'
    html += '        <tr>\n'
    html += '            <th></th>\n'
    for state in states:
        html += f'            <th>S<sub>{state}</sub></th>\n'
    html += '        </tr>\n'
    
    for i, state in enumerate(states):
        html += '        <tr>\n'
        html += f'            <th>S<sub>{state}</sub></th>\n'
        for j in range(len(states)):
            html += f'            <td>{M[i, j]:.4f}</td>\n'
        html += '        </tr>\n'
    
    html += '    </table>\n'
    html += '</div>\n'
    return html

def format_vector_html(p_bar, states, title="p̅"):
    """Форматирует вектор в HTML"""
    if title == "":
        html = '<p>['
    else:
        html = f'<p><strong>{title} = [</strong>'
    values = []
    for i, state in enumerate(states):
        values.append(f'{p_bar[i]:.6f}')
    html += '; '.join(values)
    if title == "":
        html += ']</p>\n'
    else:
        html += ']</p>\n'
    return html

def generate_html_report():
    """Генерирует HTML отчет"""
    
    # Загружаем результаты
    try:
        with open('lab10_results.pkl', 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print("Ошибка: файл lab10_results.pkl не найден. Сначала запустите lab10_calculation.py")
        return
    
    student1 = results['student1']
    student2 = results['student2']
    
    M1 = student1['M']  # M - матрица переходов
    p_bar1 = student1['p_bar']  # p̅ - вектор стационарных вероятностей
    states1 = student1['states']
    
    M2 = student2['M']
    p_bar2 = student2['p_bar']
    states2 = student2['states']
    
    html = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа 10: Марковские цепи для студентов</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
            line-height: 1.6;
            color: #000;
            background: #fff;
        }
        h1 {
            text-align: center;
            font-size: 18pt;
            margin-bottom: 20px;
            font-weight: bold;
        }
        h2 {
            font-size: 16pt;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: bold;
        }
        h3 {
            font-size: 14pt;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        p {
            margin: 10px 0;
            text-align: justify;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            border: 1px solid #000;
        }
        th, td {
            border: 1px solid #000;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #fff;
            font-weight: bold;
        }
        .formula {
            text-align: center;
            margin: 15px 0;
            font-style: italic;
        }
        .solution {
            margin: 15px 0;
        }
        .result {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #000;
        }
        .author {
            text-align: right;
            margin-bottom: 20px;
            font-size: 14pt;
        }
        .equation {
            text-align: center;
            margin: 15px 0;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>Лабораторная работа на тему «Марковские цепи для студентов»</h1>
    
    <div class="author">
        <p><strong>Выполнил:</strong> Тимошинов Егор Борисович</p>
        <p><strong>Группа:</strong> 16</p>
    </div>

    <h2>Цель работы</h2>
    <p>Для оставшихся двух студентов определить матрицу переходных вероятностей и решить векторно-матричное уравнение (10)-(11) для нахождения стационарного распределения.</p>
    
    <h2>Теоретические сведения</h2>
    <p>Для марковской цепи с матрицей переходных вероятностей M стационарное распределение p̅ определяется из системы уравнений:</p>
    <div class="equation">
        <p><strong>Уравнение (10):</strong> p̅ × (M - E) = 0</p>
        <p>где p̅ — вектор-строка установившихся состояний (p̅₁; p̅₂; p̅₃; p̅₄),</p>
        <p>M — матрица переходов, E — единичная матрица.</p>
        <p><strong>Уравнение (11):</strong> Σ<sub>j=1</sub><sup>4</sup> p̅<sub>j</sub> = 1</p>
    </div>
    <p>Уравнение (10) представляет собой систему линейных алгебраических уравнений, которая является линейно-зависимой. Уравнение (11) — нормировочное условие, используемое для дополнения системы.</p>
    
'''
    
    # Студент 1
    html += f'''
    <h2>Студент 1</h2>
    <p>Определим матрицу переходных вероятностей на основе данных о переходах между состояниями.</p>
    <p>Состояния: {', '.join([f'S<sub>{s}</sub>' for s in states1])}</p>
    
    {format_matrix_html(M1, states1, "M₁")}
    
    <h3>Решение векторно-матричного уравнения (10)-(11)</h3>
    <div class="solution">
        <p>Для нахождения стационарного распределения решаем систему уравнений:</p>
        <p><strong>Уравнение (10):</strong> p̅₁ × (M₁ - E) = 0</p>
        <p>где p̅₁ — вектор-строка установившихся состояний (p̅₁₁; p̅₁₂; p̅₁₃; p̅₁₄),</p>
        <p>M₁ — матрица переходов, E — единичная матрица.</p>
        <p><strong>Уравнение (11):</strong> Σ<sub>j=1</sub><sup>4</sup> p̅₁<sub>j</sub> = 1</p>
        
        <p>Преобразуем уравнение (10) в систему: (M₁ - E)ᵀ p̅₁ᵀ = 0.</p>
        <p>Система линейно-зависима, поэтому заменяем последнее уравнение на нормировочное условие (11).</p>
        
        <p>Решая систему, получаем стационарное распределение:</p>
        {format_vector_html(p_bar1, states1, "p̅₁")}
        
        <p>Проверка уравнения (11):</p>
        <p>Σ<sub>j=1</sub><sup>4</sup> p̅₁<sub>j</sub> = {p_bar1.sum():.6f} ≈ 1,000</p>
        
        <p>Проверка уравнения (10) p̅ × (M - E) = 0:</p>
        <p>p̅₁ × (M₁ - E) = {format_vector_html(p_bar1 @ (M1 - np.eye(len(states1))), states1, "")[3:-4]}</p>
        <p>Максимальная абсолютная величина: {np.max(np.abs(p_bar1 @ (M1 - np.eye(len(states1))))):.10f}</p>
    </div>
    
    <div class="result">
        <p><strong>Результаты для студента 1:</strong></p>
        <table>
            <tr>
                <th>Состояние</th>
                <th>Стационарная вероятность p̅₁</th>
            </tr>
'''
    for i, state in enumerate(states1):
        html += f'''            <tr>
                <td>S<sub>{state}</sub></td>
                <td>{p_bar1[i]:.6f}</td>
            </tr>
'''
    html += '''        </table>
    </div>
    
'''
    
    # Студент 2
    html += f'''
    <h2>Студент 2</h2>
    <p>Определим матрицу переходных вероятностей на основе данных о переходах между состояниями.</p>
    <p>Состояния: {', '.join([f'S<sub>{s}</sub>' for s in states2])}</p>
    
    {format_matrix_html(M2, states2, "M₂")}
    
    <h3>Решение векторно-матричного уравнения (10)-(11)</h3>
    <div class="solution">
        <p>Для нахождения стационарного распределения решаем систему уравнений:</p>
        <p><strong>Уравнение (10):</strong> p̅₂ × (M₂ - E) = 0</p>
        <p>где p̅₂ — вектор-строка установившихся состояний (p̅₂₁; p̅₂₂; p̅₂₃; p̅₂₄),</p>
        <p>M₂ — матрица переходов, E — единичная матрица.</p>
        <p><strong>Уравнение (11):</strong> Σ<sub>j=1</sub><sup>4</sup> p̅₂<sub>j</sub> = 1</p>
        
        <p>Преобразуем уравнение (10) в систему: (M₂ - E)ᵀ p̅₂ᵀ = 0.</p>
        <p>Система линейно-зависима, поэтому заменяем последнее уравнение на нормировочное условие (11).</p>
        
        <p>Решая систему, получаем стационарное распределение:</p>
        {format_vector_html(p_bar2, states2, "p̅₂")}
        
        <p>Проверка уравнения (11):</p>
        <p>Σ<sub>j=1</sub><sup>4</sup> p̅₂<sub>j</sub> = {p_bar2.sum():.6f} ≈ 1,000</p>
        
        <p>Проверка уравнения (10) p̅ × (M - E) = 0:</p>
        <p>p̅₂ × (M₂ - E) = {format_vector_html(p_bar2 @ (M2 - np.eye(len(states2))), states2, "")[3:-4]}</p>
        <p>Максимальная абсолютная величина: {np.max(np.abs(p_bar2 @ (M2 - np.eye(len(states2))))):.10f}</p>
    </div>
    
    <div class="result">
        <p><strong>Результаты для студента 2:</strong></p>
        <table>
            <tr>
                <th>Состояние</th>
                <th>Стационарная вероятность p̅₂</th>
            </tr>
'''
    for i, state in enumerate(states2):
        html += f'''            <tr>
                <td>S<sub>{state}</sub></td>
                <td>{p_bar2[i]:.6f}</td>
            </tr>
'''
    html += '''        </table>
    </div>
    
    <h2>Выводы</h2>
    <p>Для обоих студентов были определены матрицы переходных вероятностей и решены векторно-матричные уравнения (10)-(11) для нахождения стационарных распределений.</p>
    <p>Стационарное распределение показывает долгосрочные вероятности нахождения системы в каждом из состояний при бесконечном числе переходов.</p>
    
</body>
</html>'''
    
    # Сохраняем HTML
    output_file = 'Timoshinov_E_B_Lab_10.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML отчет сохранен в файл: {output_file}")

if __name__ == '__main__':
    generate_html_report()
