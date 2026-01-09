# -*- coding: utf-8 -*-
"""
Главный скрипт для выполнения лабораторной работы 7
Нечеткая классификация
"""

import os
import sys

# Определяем путь к директории скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 80)
print("Лабораторная работа 7: Нечеткая классификация")
print("Выполнил: Тимошинов Егор Борисович, группа 16")
print("=" * 80)
print()

# Шаг 1: Выполнение классификации
print("Шаг 1: Выполнение нечеткой классификации...")
print()
import fuzzy_classification_lab

print()
print("=" * 80)
print()

# Шаг 2: Генерация HTML-отчета
print("Шаг 2: Генерация HTML-отчета...")
print()
import generate_html_report

print()
print("=" * 80)
print("Лабораторная работа выполнена успешно!")
print("=" * 80)
