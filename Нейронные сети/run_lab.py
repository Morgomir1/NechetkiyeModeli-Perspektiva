import os
import sys

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Импортируем и запускаем основной скрипт
print("Запуск лабораторной работы по нейронным сетям...")
print("=" * 60)

import neural_network_lab

print("\n" + "=" * 60)
print("Генерация HTML-отчета...")

import generate_lab_report

print("\n" + "=" * 60)
print("Лабораторная работа завершена!")


