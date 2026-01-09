import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns

# Настройка для черно-белого вывода
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_style("whitegrid")

# Генерация данных для примера (заболевание в зависимости от возраста)
np.random.seed(42)
n_samples = 100
ages = np.random.randint(25, 85, n_samples)
# Вероятность заболевания увеличивается с возрастом
prob_disease = 1 / (1 + np.exp(-(ages - 50) / 10))
disease = (np.random.random(n_samples) < prob_disease).astype(int)

# Создание DataFrame
data = pd.DataFrame({
    'age': ages,
    'disease': disease
})

# Сортировка данных по возрасту для лучшей визуализации
data = data.sort_values('age').reset_index(drop=True)

print("Первые 20 строк данных:")
print(data.head(20))
print("\nСтатистика по данным:")
print(data.describe())
print(f"\nКоличество случаев заболевания: {data['disease'].sum()}")
print(f"Количество случаев без заболевания: {(data['disease'] == 0).sum()}")

# Подготовка данных для модели
X = data[['age']].values
y = data['disease'].values

# Построение модели логистической регрессии
model = LogisticRegression()
model.fit(X, y)

# Получение предсказанных вероятностей
y_pred_proba = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)

# Вывод коэффициентов модели
print("\nКоэффициенты модели:")
print(f"Intercept (β₀): {model.intercept_[0]:.4f}")
print(f"Coefficient (β₁): {model.coef_[0][0]:.4f}")

# Вычисление ROC-кривой
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nAUC (Area Under Curve): {roc_auc:.4f}")

# Матрица ошибок
cm = confusion_matrix(y, y_pred)
print("\nМатрица ошибок:")
print(cm)

# Классификационный отчет
print("\nКлассификационный отчет:")
print(classification_report(y, y_pred))

# Построение графиков

# 1. График зависимости вероятности заболевания от возраста
fig, ax = plt.subplots(figsize=(10, 6))
age_range = np.linspace(25, 85, 100).reshape(-1, 1)
prob_range = model.predict_proba(age_range)[:, 1]

# Показываем фактические значения (0/1) и предсказанные вероятности
ax.scatter(data['age'], data['disease'], alpha=0.3, s=30, color='gray', marker='o', label='Фактические значения')
ax.scatter(data['age'], y_pred_proba, alpha=0.6, s=40, color='black', marker='x', label='Предсказанные вероятности')
ax.plot(age_range, prob_range, 'k-', linewidth=2, label='Логистическая кривая')
ax.set_xlabel('Возраст', fontsize=12)
ax.set_ylabel('Вероятность заболевания', fontsize=12)
ax.set_title('Зависимость вероятности заболевания от возраста', fontsize=14)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_ylim([-0.05, 1.05])
plt.tight_layout()
plt.savefig('logistic_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. ROC-кривая
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, 'k-', linewidth=2, label=f'ROC кривая (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Случайный классификатор')
ax.set_xlabel('Доля False Positive (FP)', fontsize=12)
ax.set_ylabel('Доля True Positive (TP)', fontsize=12)
ax.set_title('ROC-кривая', fontsize=14)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Матрица ошибок
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='gray', cbar=False, ax=ax, 
            xticklabels=['Нет заболевания', 'Есть заболевание'],
            yticklabels=['Нет заболевания', 'Есть заболевание'])
ax.set_xlabel('Предсказанный класс', fontsize=12)
ax.set_ylabel('Истинный класс', fontsize=12)
ax.set_title('Матрица ошибок', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Распределение предсказанных вероятностей
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_proba[data['disease'] == 0], bins=20, alpha=0.7, color='gray', 
        label='Нет заболевания', edgecolor='black')
ax.hist(y_pred_proba[data['disease'] == 1], bins=20, alpha=0.7, color='black', 
        label='Есть заболевание', edgecolor='black')
ax.set_xlabel('Предсказанная вероятность', fontsize=12)
ax.set_ylabel('Частота', fontsize=12)
ax.set_title('Распределение предсказанных вероятностей', fontsize=14)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('probability_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Сохранение результатов
results = {
    'model': model,
    'fpr': fpr,
    'tpr': tpr,
    'roc_auc': roc_auc,
    'confusion_matrix': cm,
    'data': data,
    'y_pred_proba': y_pred_proba,
    'y_pred': y_pred,
    'coefficients': {
        'intercept': model.intercept_[0],
        'coefficient': model.coef_[0][0]
    }
}

import pickle
with open('logistic_regression_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Сохранение данных в CSV
data_with_predictions = data.copy()
data_with_predictions['predicted_probability'] = y_pred_proba
data_with_predictions['predicted_class'] = y_pred
data_with_predictions.to_csv('logistic_regression_data.csv', index=False, encoding='utf-8-sig')

print("\nВсе графики и результаты сохранены!")
print("Файлы:")
print("- logistic_curve.png")
print("- roc_curve.png")
print("- confusion_matrix.png")
print("- probability_distribution.png")
print("- logistic_regression_results.pkl")
print("- logistic_regression_data.csv")

