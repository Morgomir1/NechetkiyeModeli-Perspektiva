import os
import sys

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Лабораторная работа: Регрессионный анализ")
print("Выполнил: Тимошинов Егор Борисович, группа 16")
print("=" * 60)

# Шаг 1: Выполнение анализа
print("\n[1/2] Выполнение регрессионного анализа...")
print("-" * 60)
exec(open('regression_analysis_lab.py', encoding='utf-8').read())

# Шаг 2: Генерация HTML отчета
print("\n[2/2] Генерация HTML отчета...")
print("-" * 60)
exec(open('generate_lab_report.py', encoding='utf-8').read())

print("\n" + "=" * 60)
print("Лабораторная работа успешно завершена!")
print("Результаты сохранены в файле: Timoshinov_E_B_Lab_RegressionAnalysis.html")
print("=" * 60)
