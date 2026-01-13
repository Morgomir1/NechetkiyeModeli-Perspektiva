# -*- coding: utf-8 -*-
"""
Главный скрипт для выполнения лабораторной работы 10
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА 10")
print("Определение матрицы переходных вероятностей и решение векторно-матричного уравнения")
print("=" * 70)

# Шаг 1: Выполнение вычислений
print("\nШаг 1: Выполнение вычислений...")
try:
    import lab10_calculation
    print("✓ Вычисления выполнены успешно")
except Exception as e:
    print(f"✗ Ошибка при выполнении вычислений: {e}")
    sys.exit(1)

# Шаг 2: Генерация HTML отчета
print("\nШаг 2: Генерация HTML отчета...")
try:
    import generate_lab_report
    generate_lab_report.generate_html_report()
    print("✓ HTML отчет сгенерирован успешно")
except Exception as e:
    print(f"✗ Ошибка при генерации отчета: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("Лабораторная работа выполнена!")
print("Результаты сохранены в файл: Timoshinov_E_B_Lab_10.html")
print("=" * 70)
