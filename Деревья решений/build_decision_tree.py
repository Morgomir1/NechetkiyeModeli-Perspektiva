import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Генерация синтетических данных о кредитах
# Атрибуты: возраст, доход, кредитная история, цель кредита, срок кредита
np.random.seed(42)

n_samples = 1000

# Генерация признаков
age = np.random.randint(18, 70, n_samples)
income = np.random.randint(20000, 150000, n_samples)
credit_history = np.random.choice(['хорошая', 'средняя', 'плохая'], n_samples)
loan_purpose = np.random.choice(['потребительский', 'автомобиль', 'недвижимость', 'бизнес'], n_samples)
loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)

# Создание целевой переменной (возврат кредита: Да/Нет)
# Логика: вероятность возврата зависит от признаков
def generate_target(age, income, credit_history, loan_purpose, loan_term):
    prob = 0.5
    
    # Возраст: оптимальный 30-50 лет
    if 30 <= age <= 50:
        prob += 0.15
    elif age < 25 or age > 60:
        prob -= 0.1
    
    # Доход: чем выше, тем лучше
    if income > 80000:
        prob += 0.2
    elif income < 40000:
        prob -= 0.15
    
    # Кредитная история
    if credit_history == 'хорошая':
        prob += 0.25
    elif credit_history == 'средняя':
        prob += 0.05
    else:
        prob -= 0.2
    
    # Цель кредита
    if loan_purpose == 'недвижимость':
        prob += 0.1
    elif loan_purpose == 'бизнес':
        prob -= 0.1
    
    # Срок кредита: короткие лучше
    if loan_term <= 24:
        prob += 0.1
    elif loan_term >= 48:
        prob -= 0.1
    
    return 'Да' if np.random.random() < prob else 'Нет'

target = [generate_target(age[i], income[i], credit_history[i], loan_purpose[i], loan_term[i]) 
          for i in range(n_samples)]

# Создание DataFrame
data = pd.DataFrame({
    'возраст': age,
    'доход': income,
    'кредитная_история': credit_history,
    'цель_кредита': loan_purpose,
    'срок_кредита': loan_term,
    'возврат_кредита': target
})

# Кодирование категориальных переменных
from sklearn.preprocessing import LabelEncoder

le_history = LabelEncoder()
le_purpose = LabelEncoder()
le_target = LabelEncoder()

data['кредитная_история_код'] = le_history.fit_transform(data['кредитная_история'])
data['цель_кредита_код'] = le_purpose.fit_transform(data['цель_кредита'])
data['возврат_кредита_код'] = le_target.fit_transform(data['возврат_кредита'])

# Подготовка данных для обучения
X = data[['возраст', 'доход', 'кредитная_история_код', 'цель_кредита_код', 'срок_кредита']]
y = data['возврат_кредита_код']

# Разделение на обучающую и тестовую выборки (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Построение дерева решений
tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=5,
    random_state=42
)

tree.fit(X_train, y_train)

# Функция для конвертации matplotlib figure в base64
def fig_to_base64(fig):
    """Конвертирует matplotlib figure в base64 строку"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# Задание 1: Визуализация данных
# Распределение количественных признаков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

axes[0, 0].hist(data['возраст'], bins=20, color=colors[0], edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Возраст', fontsize=11)
axes[0, 0].set_ylabel('Частота', fontsize=11)
axes[0, 0].set_title('Распределение возраста клиентов', fontsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.3)

axes[0, 1].hist(data['доход'], bins=20, color=colors[1], edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Доход', fontsize=11)
axes[0, 1].set_ylabel('Частота', fontsize=11)
axes[0, 1].set_title('Распределение дохода клиентов', fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.3)

axes[1, 0].hist(data['срок_кредита'], bins=5, color=colors[2], edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Срок кредита (месяцы)', fontsize=11)
axes[1, 0].set_ylabel('Частота', fontsize=11)
axes[1, 0].set_title('Распределение срока кредита', fontsize=12)
axes[1, 0].grid(True, linestyle='--', alpha=0.3)

# Распределение целевой переменной
target_counts = data['возврат_кредита'].value_counts()
axes[1, 1].bar(target_counts.index, target_counts.values, color=[colors[2], colors[3]], edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Возврат кредита', fontsize=11)
axes[1, 1].set_ylabel('Количество', fontsize=11)
axes[1, 1].set_title('Распределение целевой переменной', fontsize=12)
axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
img_data_dist = fig_to_base64(fig)

# Распределение категориальных признаков
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

history_counts = data['кредитная_история'].value_counts()
axes[0].bar(history_counts.index, history_counts.values, color=colors[:3], edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Кредитная история', fontsize=11)
axes[0].set_ylabel('Количество', fontsize=11)
axes[0].set_title('Распределение кредитной истории', fontsize=12)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

purpose_counts = data['цель_кредита'].value_counts()
axes[1].bar(purpose_counts.index, purpose_counts.values, color=colors, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Цель кредита', fontsize=11)
axes[1].set_ylabel('Количество', fontsize=11)
axes[1].set_title('Распределение цели кредита', fontsize=12)
axes[1].grid(True, axis='y', linestyle='--', alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
img_cat_dist = fig_to_base64(fig)

# Корреляционная матрица количественных признаков
fig, ax = plt.subplots(figsize=(10, 8))
numeric_data = data[['возраст', 'доход', 'срок_кредита', 'возврат_кредита_код']]
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Корреляционная матрица количественных признаков', fontsize=14, pad=20)
plt.tight_layout()
img_corr = fig_to_base64(fig)

# Статистика по данным
data_stats = data[['возраст', 'доход', 'срок_кредита']].describe()

# Визуализация дерева решений
fig, ax = plt.subplots(figsize=(20, 12))
feature_names = ['Возраст', 'Доход', 'Кредитная история', 'Цель кредита', 'Срок кредита']
class_names = ['Нет', 'Да']
plot_tree(tree, filled=True, feature_names=feature_names, class_names=class_names, 
          ax=ax, fontsize=10, rounded=True)
ax.set_title('Дерево решений для классификации возврата кредита', fontsize=16, pad=20)
plt.tight_layout()
img_tree = fig_to_base64(fig)

# Текстовая визуализация дерева
tree_text = export_text(tree, feature_names=feature_names, max_depth=5)

# Предсказания на тестовой выборке
y_pred = tree.predict(X_test)
y_pred_proba = tree.predict_proba(X_test)

# Точность модели
accuracy = accuracy_score(y_test, y_pred)

# Таблица сопряженности
cm = confusion_matrix(y_test, y_pred)

# Визуализация таблицы сопряженности
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['Нет', 'Да'], yticklabels=['Нет', 'Да'])
ax.set_xlabel('Предсказанный класс', fontsize=12)
ax.set_ylabel('Истинный класс', fontsize=12)
ax.set_title('Таблица сопряженности', fontsize=14)
plt.tight_layout()
img_cm = fig_to_base64(fig)

# Статистика по таблице сопряженности
tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp
correct = tn + tp

# Отчет о классификации
class_report = classification_report(y_test, y_pred, target_names=['Нет', 'Да'], output_dict=True)

# Важность признаков
feature_importance = pd.DataFrame({
    'Признак': feature_names,
    'Важность': tree.feature_importances_
}).sort_values('Важность', ascending=False)

# Визуализация важности признаков
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = ax.barh(feature_importance['Признак'], feature_importance['Важность'], color=colors)
ax.set_xlabel('Важность признака', fontsize=12)
ax.set_title('Важность признаков в дереве решений', fontsize=14)
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
for i, (idx, row) in enumerate(feature_importance.iterrows()):
    ax.text(row['Важность'] + 0.01, i, f"{row['Важность']:.3f}", 
            va='center', fontsize=10)
plt.tight_layout()
img_importance = fig_to_base64(fig)

# Создание HTML отчета
html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лабораторная работа по деревьям решений</title>
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
            font-size: 18pt;
        }}
        h2 {{
            text-align: center;
            margin-bottom: 30px;
            font-weight: normal;
            font-size: 16pt;
        }}
        h3 {{
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 14pt;
        }}
        p {{
            text-align: justify;
            line-height: 1.6;
            margin-bottom: 15px;
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
            text-align: center;
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
            font-size: 11px;
            overflow-x: auto;
            margin: 15px 0;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="author">
        <p><strong>Выполнил:</strong> Тимошинов Егор Борисович</p>
        <p><strong>Группа:</strong> 16</p>
    </div>
    
    <h1>Лабораторная работа</h1>
    <h2>Деревья решений для классификации</h2>
    
    <h3>Цель работы</h3>
    <p>Изучение методов построения деревьев решений для решения задач классификации. Построение дерева решений для предсказания возврата кредита на основе характеристик клиента.</p>
    
    <h3>Задание</h3>
    <ol>
        <li>Подготовить данные о клиентах банка для построения дерева решений. Выполнить исследовательский анализ данных и визуализацию распределения признаков.</li>
        <li>Построить дерево решений для классификации возврата кредита с использованием алгоритма CART.</li>
        <li>Проанализировать важность признаков в построенном дереве решений.</li>
        <li>Выполнить классификацию на тестовой выборке и построить таблицу сопряженности.</li>
        <li>Оценить точность модели с помощью различных метрик классификации.</li>
    </ol>
    
    <h2>Результаты выполнения задания</h2>
    
    <h3>Задание 1. Подготовка и анализ данных</h3>
    <p>Для построения модели использовался набор данных о клиентах банка, включающий следующие признаки:</p>
    <ul>
        <li>Возраст клиента (количественный признак)</li>
        <li>Доход клиента (количественный признак)</li>
        <li>Кредитная история (категориальный признак: хорошая, средняя, плохая)</li>
        <li>Цель кредита (категориальный признак: потребительский, автомобиль, недвижимость, бизнес)</li>
        <li>Срок кредита в месяцах (количественный признак)</li>
    </ul>
    <p>Целевая переменная: возврат кредита (Да/Нет).</p>
    <p>Общее количество наблюдений: {n_samples}. Данные разделены на обучающую выборку (70%) и тестовую выборку (30%).</p>
    
    <p><strong>Описательная статистика количественных признаков:</strong></p>
    <table>
        <tr>
            <th>Признак</th>
            <th>Среднее</th>
            <th>Стандартное отклонение</th>
            <th>Минимум</th>
            <th>Максимум</th>
        </tr>
        <tr>
            <td>Возраст</td>
            <td>{data_stats.loc['mean', 'возраст']:.2f}</td>
            <td>{data_stats.loc['std', 'возраст']:.2f}</td>
            <td>{data_stats.loc['min', 'возраст']:.0f}</td>
            <td>{data_stats.loc['max', 'возраст']:.0f}</td>
        </tr>
        <tr>
            <td>Доход</td>
            <td>{data_stats.loc['mean', 'доход']:.2f}</td>
            <td>{data_stats.loc['std', 'доход']:.2f}</td>
            <td>{data_stats.loc['min', 'доход']:.0f}</td>
            <td>{data_stats.loc['max', 'доход']:.0f}</td>
        </tr>
        <tr>
            <td>Срок кредита</td>
            <td>{data_stats.loc['mean', 'срок_кредита']:.2f}</td>
            <td>{data_stats.loc['std', 'срок_кредита']:.2f}</td>
            <td>{data_stats.loc['min', 'срок_кредита']:.0f}</td>
            <td>{data_stats.loc['max', 'срок_кредита']:.0f}</td>
        </tr>
    </table>
    
    <p>На рисунках ниже представлены распределения признаков и целевой переменной:</p>
    <img src="data:image/png;base64,{img_data_dist}" alt="Распределение количественных признаков и целевой переменной">
    <p><em>Рисунок 1. Распределение количественных признаков и целевой переменной</em></p>
    
    <img src="data:image/png;base64,{img_cat_dist}" alt="Распределение категориальных признаков">
    <p><em>Рисунок 2. Распределение категориальных признаков</em></p>
    
    <img src="data:image/png;base64,{img_corr}" alt="Корреляционная матрица">
    <p><em>Рисунок 3. Корреляционная матрица количественных признаков</em></p>
    
    <h3>Задание 2. Построение дерева решений</h3>
    <p>Для построения дерева решений использовался алгоритм CART (Classification and Regression Trees) с критерием разделения Gini. Параметры модели:</p>
    <ul>
        <li>Максимальная глубина дерева: 5</li>
        <li>Минимальное количество образцов для разделения узла: 20</li>
        <li>Минимальное количество образцов в листе: 5</li>
    </ul>
    <p>Построенное дерево решений представлено на рисунке ниже.</p>
    <img src="data:image/png;base64,{img_tree}" alt="Дерево решений">
    <p><em>Рисунок 4. Визуализация дерева решений</em></p>
    
    <h3>Задание 3. Анализ важности признаков</h3>
    <p>Важность признаков в построенном дереве решений:</p>
    <table>
        <tr>
            <th>Признак</th>
            <th>Важность</th>
        </tr>
"""
for _, row in feature_importance.iterrows():
    html_content += f"""        <tr>
            <td>{row['Признак']}</td>
            <td>{row['Важность']:.4f}</td>
        </tr>
"""
html_content += f"""    </table>
    <img src="data:image/png;base64,{img_importance}" alt="Важность признаков">
    <p><em>Рисунок 5. Важность признаков в дереве решений</em></p>
    
    <h3>Задание 4. Классификация на тестовой выборке</h3>
    <p>Для оценки качества построенной модели была выполнена классификация на тестовой выборке, содержащей {len(y_test)} наблюдений.</p>
    
    <h3>Задание 5. Таблица сопряженности и оценка точности</h3>
    <p>Таблица сопряженности показывает количество правильно и неправильно классифицированных наблюдений:</p>
    <table>
        <tr>
            <th></th>
            <th>Предсказано: Нет</th>
            <th>Предсказано: Да</th>
        </tr>
        <tr>
            <th>Истинно: Нет</th>
            <td>{tn}</td>
            <td>{fp}</td>
        </tr>
        <tr>
            <th>Истинно: Да</th>
            <td>{fn}</td>
            <td>{tp}</td>
        </tr>
    </table>
    <img src="data:image/png;base64,{img_cm}" alt="Таблица сопряженности">
    <p><em>Рисунок 6. Таблица сопряженности</em></p>
    <p>Характеристики точности построенной модели:</p>
    <table>
        <tr>
            <th>Метрика</th>
            <th>Значение</th>
        </tr>
        <tr>
            <td>Точность (Accuracy)</td>
            <td>{accuracy:.4f} ({accuracy*100:.2f}%)</td>
        </tr>
        <tr>
            <td>Правильно классифицировано</td>
            <td>{correct} из {total}</td>
        </tr>
        <tr>
            <td>Ошибок классификации</td>
            <td>{total - correct} из {total}</td>
        </tr>
    </table>
    
    <h3>Детальный отчет о классификации</h3>
    <table>
        <tr>
            <th>Класс</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-score</th>
            <th>Support</th>
        </tr>
        <tr>
            <td>Нет</td>
            <td>{class_report['Нет']['precision']:.4f}</td>
            <td>{class_report['Нет']['recall']:.4f}</td>
            <td>{class_report['Нет']['f1-score']:.4f}</td>
            <td>{int(class_report['Нет']['support'])}</td>
        </tr>
        <tr>
            <td>Да</td>
            <td>{class_report['Да']['precision']:.4f}</td>
            <td>{class_report['Да']['recall']:.4f}</td>
            <td>{class_report['Да']['f1-score']:.4f}</td>
            <td>{int(class_report['Да']['support'])}</td>
        </tr>
        <tr>
            <td><strong>Среднее (macro avg)</strong></td>
            <td>{class_report['macro avg']['precision']:.4f}</td>
            <td>{class_report['macro avg']['recall']:.4f}</td>
            <td>{class_report['macro avg']['f1-score']:.4f}</td>
            <td>{int(class_report['macro avg']['support'])}</td>
        </tr>
        <tr>
            <td><strong>Среднее (weighted avg)</strong></td>
            <td>{class_report['weighted avg']['precision']:.4f}</td>
            <td>{class_report['weighted avg']['recall']:.4f}</td>
            <td>{class_report['weighted avg']['f1-score']:.4f}</td>
            <td>{int(class_report['weighted avg']['support'])}</td>
        </tr>
    </table>
    
    <h3>Текстовая структура дерева</h3>
    <p>Ниже представлена текстовая структура построенного дерева решений (первые уровни):</p>
    <div class="code-block">{tree_text[:2000]}...</div>
    
    <h3>Выводы</h3>
    <p>В ходе выполнения лабораторной работы было построено дерево решений для классификации возврата кредита. Модель показала точность {accuracy*100:.2f}% на тестовой выборке. Наиболее важными признаками для классификации оказались: {feature_importance.iloc[0]['Признак']} (важность {feature_importance.iloc[0]['Важность']:.4f}) и {feature_importance.iloc[1]['Признак']} (важность {feature_importance.iloc[1]['Важность']:.4f}).</p>
    <p>Из {total} наблюдений в тестовой выборке модель правильно классифицировала {correct} наблюдений. Таблица сопряженности показывает, что модель лучше предсказывает класс "Нет" (не возврат кредита), чем класс "Да" (возврат кредита).</p>
    <p>Построенное дерево решений может быть использовано банком для оценки риска выдачи кредита новым клиентам на основе их характеристик.</p>
</body>
</html>
"""

# Сохранение HTML файла
with open('Timoshinov_E_B_Lab_DecisionTree.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML отчет создан: Timoshinov_E_B_Lab_DecisionTree.html")
print(f"Точность модели: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Правильно классифицировано: {correct} из {total}")

