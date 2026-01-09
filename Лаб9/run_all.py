# -*- coding: utf-8 -*-
import os
import sys

# Настройка кодировки для Windows консоли через переменную окружения
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Меняем рабочую директорию на директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Лабораторная работа: Система Такаги-Сугено и ANFIS")
print("=" * 60)

# Шаг 1: Построение системы Такаги-Сугено
print("\n[1/4] Построение системы Такаги-Сугено для Credit...")
try:
    from sugeno_credit_system import create_credit_sugeno_system, generate_credit_data
    import pickle
    
    credit_data, le_history, le_purpose = generate_credit_data()
    sugeno_system = create_credit_sugeno_system()
    
    # Тестирование
    test_cases = [
        {'возраст': 25, 'доход': 30000},
        {'возраст': 35, 'доход': 70000},
        {'возраст': 45, 'доход': 100000},
        {'возраст': 55, 'доход': 50000},
        {'возраст': 30, 'доход': 120000},
    ]
    
    results = []
    for test in test_cases:
        output = sugeno_system.infer(test)
        results.append({
            'возраст': test['возраст'],
            'доход': test['доход'],
            'оценка': output
        })
    
    # Сохраняем только результаты, без системы (система содержит функции, которые нельзя сериализовать)
    with open('sugeno_credit_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'test_cases': test_cases,
            'data': credit_data
            # 'system': sugeno_system,  # Не сохраняем систему, т.к. содержит функции
        }, f)
    
    # Сохраняем систему отдельно для использования в generate_plots
    # Создадим её заново при необходимости
    
    print("✓ Система Такаги-Сугено построена и сохранена")
except Exception as e:
    print(f"✗ Ошибка при построении системы Сугено: {e}")
    sys.exit(1)

# Шаг 2: Обучение ANFIS
print("\n[2/4] Обучение ANFIS...")
try:
    import subprocess
    # Устанавливаем кодировку через переменную окружения для subprocess
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run([sys.executable, 'anfis_training.py'], 
                          capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    if result.returncode == 0:
        print("✓ ANFIS обучен успешно")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ Ошибка при обучении ANFIS: {result.stderr}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Ошибка при обучении ANFIS: {e}")
    sys.exit(1)

# Шаг 3: Генерация графиков
print("\n[3/4] Генерация графиков...")
try:
    from generate_plots import generate_all_plots
    images = generate_all_plots()
    print(f"✓ Сгенерировано {len(images)} графиков")
except Exception as e:
    print(f"✗ Ошибка при генерации графиков: {e}")
    sys.exit(1)

# Шаг 4: Генерация HTML отчета
print("\n[4/4] Генерация HTML отчета...")
try:
    import subprocess
    # Устанавливаем кодировку через переменную окружения для subprocess
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run([sys.executable, 'generate_lab_report.py'], 
                          capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    if result.returncode == 0:
        print("✓ HTML отчет создан")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ Ошибка при генерации отчета: {result.stderr}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Ошибка при генерации отчета: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Лабораторная работа выполнена успешно!")
print("=" * 60)
print("\nСозданные файлы:")
print("  - sugeno_credit_results.pkl")
print("  - anfis_results.pkl")
print("  - plot_images.pkl")
print("  - Timoshinov_E_B_Lab_9.html")
print("\nОткройте файл Timoshinov_E_B_Lab_9.html в браузере для просмотра результатов.")
