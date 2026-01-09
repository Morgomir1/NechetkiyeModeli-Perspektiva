# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска лабораторной работы 10
"""

import os
import sys

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Лабораторная работа 10: Сравнение методов Мамдани и Сугено")
print("=" * 60)

# Шаг 1: Выполнение основной лабораторной работы
print("\nШаг 1: Выполнение нечетких систем...")
try:
    import fuzzy_comparison_lab
    print("✓ Нечеткие системы успешно выполнены")
except Exception as e:
    print(f"✗ Ошибка при выполнении нечетких систем: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Шаг 2: Генерация графиков
print("\nШаг 2: Генерация графиков...")
try:
    import generate_plots
    print("✓ Графики успешно сгенерированы")
except Exception as e:
    print(f"✗ Ошибка при генерации графиков: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Шаг 3: Генерация HTML отчета
print("\nШаг 3: Генерация HTML отчета...")
try:
    import generate_lab_report
    print("✓ HTML отчет успешно создан")
except Exception as e:
    print(f"✗ Ошибка при генерации HTML отчета: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Лабораторная работа выполнена успешно!")
print("Откройте файл Timoshinov_E_B_Lab_10.html в браузере")
print("=" * 60)
