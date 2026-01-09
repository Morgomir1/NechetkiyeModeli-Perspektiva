#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуска всех этапов лабораторной работы
"""
import os
import sys

# Установка рабочей директории
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Запуск лабораторной работы по системам Суджено")
print("=" * 60)

# Шаг 1: Запуск системы Суджено
print("\n1. Запуск системы Суджено...")
try:
    import sugeno_fuzzy_system
    print("   ✓ Система Суджено успешно создана и протестирована")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    sys.exit(1)

# Шаг 2: Генерация графиков
print("\n2. Генерация графиков...")
try:
    import generate_plots
    print("   ✓ Графики успешно сгенерированы")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Шаг 3: Генерация HTML отчета
print("\n3. Генерация HTML отчета...")
try:
    import generate_lab_report
    print("   ✓ HTML отчет успешно сгенерирован")
except Exception as e:
    print(f"   ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Все этапы выполнены успешно!")
print("=" * 60)
print("\nСозданные файлы:")
print("  - sugeno_results.pkl (результаты работы системы)")
print("  - temperature_membership.png (график функций принадлежности для температуры)")
print("  - dT_membership.png (график функций принадлежности для скорости изменения)")
print("  - sugeno_surface.png (поверхность вывода системы)")
print("  - test_results.png (результаты тестирования)")
print("  - Timoshinov_E_B_Lab_Sugeno.html (HTML отчет)")
