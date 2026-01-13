# -*- coding: utf-8 -*-
"""
Лабораторная работа 1: Перспективные информационные технологии
Повторение заданий из PDF на Python
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Настройка matplotlib для цветных графиков
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_style("whitegrid")

print("="*60)
print("Лабораторная работа 1: Перспективные информационные технологии")
print("="*60)

# 1. Загрузка данных
print("\n1. Загрузка данных...")
df = pd.read_csv('К ПЗ множ регр.csv', encoding='utf-8', sep=';', decimal=',')
print(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
print("\nПервые строки данных:")
print(df.head())

# 2. Структура таблицы
print("\n2. Структура таблицы:")
print(df.info())
print("\nТипы данных:")
print(df.dtypes)

# 3. Построение гистограммы по переменной Y
print("\n3. Построение гистограмм для переменной Y...")

# Расчет количества интервалов (формула Стерджеса)
k = round(1 + 3.322 * np.log10(len(df['Y'])))
h = (df['Y'].max() - df['Y'].min()) / k

# Гистограмма 1: простая
plt.figure(figsize=(10, 6))
plt.hist(df['Y'], bins=k, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Гистограмма частот', fontsize=14, fontweight='bold')
plt.xlabel('Y', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histogram_Y.png', dpi=300, bbox_inches='tight')
plt.close()

# Гистограмма 2: с заданными границами
plt.figure(figsize=(10, 6))
bins = np.linspace(df['Y'].min(), df['Y'].max(), k+1)
plt.hist(df['Y'], bins=bins, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Гистограмма', fontsize=14, fontweight='bold')
plt.xlabel('Y', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histogram_Y_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Описательные статистики
print("\n4. Описательные статистики...")

# Базовые статистики
summary_stats = {
    'mean': df['Y'].mean(),
    'median': df['Y'].median(),
    'min': df['Y'].min(),
    'max': df['Y'].max(),
    'range': df['Y'].max() - df['Y'].min(),
    'var': df['Y'].var(),
    'std': df['Y'].std(),
    'q25': df['Y'].quantile(0.25),
    'q50': df['Y'].quantile(0.50),
    'q75': df['Y'].quantile(0.75),
    'iqr': df['Y'].quantile(0.75) - df['Y'].quantile(0.25),
    'skewness': df['Y'].skew(),
    'kurtosis': df['Y'].kurtosis(),
    'cv': (df['Y'].std() / df['Y'].mean()) * 100  # коэффициент вариации
}

# Квантили от 0 до 1 с шагом 0.20
quantiles_20 = [df['Y'].quantile(q) for q in np.arange(0, 1.01, 0.20)]

# Стандартная ошибка среднего
se_mean = df['Y'].std() / np.sqrt(len(df['Y']))

# Доверительный интервал для среднего (99%)
confidence_level = 0.99
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha/2, len(df['Y']) - 1)
ci_lower = df['Y'].mean() - t_critical * se_mean
ci_upper = df['Y'].mean() + t_critical * se_mean

print(f"Среднее: {summary_stats['mean']:.4f}")
print(f"Медиана: {summary_stats['median']:.4f}")
print(f"Стандартное отклонение: {summary_stats['std']:.4f}")
print(f"Дисперсия: {summary_stats['var']:.4f}")
print(f"Коэффициент вариации: {summary_stats['cv']:.2f}%")
print(f"Асимметрия: {summary_stats['skewness']:.4f}")
print(f"Эксцесс: {summary_stats['kurtosis']:.4f}")

# 5. Ящики с усами
print("\n5. Построение ящиков с усами...")

# Ящик с усами для Y
plt.figure(figsize=(8, 6))
plt.boxplot(df['Y'], vert=True)
plt.title('Ящик с усами для Y', fontsize=14, fontweight='bold')
plt.ylabel('Y', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('boxplot_Y.png', dpi=300, bbox_inches='tight')
plt.close()

# Ящик с усами для X1
plt.figure(figsize=(8, 6))
plt.boxplot(df['X1'], vert=True)
plt.title('Ящик с усами для X1', fontsize=14, fontweight='bold')
plt.ylabel('X1', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('boxplot_X1.png', dpi=300, bbox_inches='tight')
plt.close()

# Горизонтальный ящик с усами для Y с точками
plt.figure(figsize=(10, 6))
plt.boxplot(df['Y'], vert=False)
plt.scatter(df['Y'], [1]*len(df['Y']), alpha=0.6, s=30, color='blue', zorder=3)
plt.title('Ящик с усами для Y (горизонтальный)', fontsize=14, fontweight='bold')
plt.xlabel('Y', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('boxplot_Y_horizontal.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Диаграммы рассеяния
print("\n6. Построение диаграмм рассеяния...")

# Диаграмма рассеяния Y от X1
plt.figure(figsize=(10, 6))
plt.scatter(df['X1'], df['Y'], color='#1b98e0', s=60, alpha=0.7)
plt.title('Диаграмма рассеяния', fontsize=14, fontweight='bold')
plt.xlabel('X1', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_X1_Y.png', dpi=300, bbox_inches='tight')
plt.close()

# Все возможные диаграммы рассеяния (pairs plot)
fig, axes = plt.subplots(len(df.columns), len(df.columns), figsize=(16, 16))
for i, col1 in enumerate(df.columns):
    for j, col2 in enumerate(df.columns):
        ax = axes[i, j]
        if i == j:
            ax.hist(df[col1], bins=15, color='lightblue', edgecolor='black', alpha=0.7)
            ax.set_title(col1, fontsize=10)
        else:
            ax.scatter(df[col2], df[col1], s=20, alpha=0.6, color='#1b98e0')
        if i == len(df.columns) - 1:
            ax.set_xlabel(col2, fontsize=8)
        if j == 0:
            ax.set_ylabel(col1, fontsize=8)
        ax.tick_params(labelsize=6)
plt.suptitle('Матрица диаграмм рассеяния', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('pairs_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Ковариация и корреляция
print("\n7. Расчет ковариации и корреляции...")

# Ковариация Y и X1
cov_Y_X1 = np.cov(df['Y'], df['X1'])[0, 1]
print(f"Ковариация Y и X1: {cov_Y_X1:.4f}")

# Коэффициент корреляции Y и X1
corr_Y_X1 = np.corrcoef(df['Y'], df['X1'])[0, 1]
print(f"Коэффициент корреляции Y и X1: {corr_Y_X1:.4f}")

# Корреляционная матрица
correlation_matrix = df.corr()
print("\nКорреляционная матрица:")
print(correlation_matrix)

# Тепловая карта корреляционной матрицы
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Тепловая карта корреляционной матрицы', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Сохранение всех результатов
results = {
    'data': df,
    'summary_stats': summary_stats,
    'quantiles_20': quantiles_20,
    'se_mean': se_mean,
    'confidence_interval_99': (ci_lower, ci_upper),
    'cov_Y_X1': cov_Y_X1,
    'corr_Y_X1': corr_Y_X1,
    'correlation_matrix': correlation_matrix,
    'k_bins': k,
    'h_binwidth': h
}

with open('lab1_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "="*60)
print("Все вычисления завершены!")
print("Результаты сохранены в lab1_results.pkl")
print("="*60)
