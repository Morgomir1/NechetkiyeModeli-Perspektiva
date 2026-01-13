# -*- coding: utf-8 -*-
"""Проверка импортов"""

try:
    import numpy as np
    print("✓ numpy установлен")
except ImportError:
    print("✗ numpy не установлен. Установите: pip install numpy")
    exit(1)

try:
    import pickle
    print("✓ pickle доступен (встроенный модуль)")
except ImportError:
    print("✗ pickle недоступен")
    exit(1)

print("\nВсе необходимые модули доступны!")
print("Можно запускать lab10_calculation.py")
