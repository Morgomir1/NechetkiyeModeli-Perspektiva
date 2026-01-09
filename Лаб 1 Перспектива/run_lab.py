import os
import sys

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Запуск анализа множественной регрессии...")
print("=" * 50)

# Запуск основного скрипта
print("\nШаг 1: Выполнение анализа...")
exec(open('multiple_regression_lab.py', encoding='utf-8').read())

print("\n" + "=" * 50)
print("\nШаг 2: Генерация HTML отчета...")
exec(open('generate_lab_report.py', encoding='utf-8').read())

print("\n" + "=" * 50)
print("\nЛабораторная работа выполнена успешно!")
print("Результаты сохранены в файле: Timoshinov_E_B_Lab_MultipleRegression.html")
