import os
import sys

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Импортируем и запускаем основной скрипт
print("Запуск лабораторной работы по логистической регрессии...")
import logistic_regression_lab

print("\nГенерация HTML отчета...")
import generate_lab_report

print("\nЛабораторная работа завершена!")
print("Откройте файл Timoshinov_E_B_Lab_LogisticRegression.html в браузере.")

