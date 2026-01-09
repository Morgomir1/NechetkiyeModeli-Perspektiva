# -*- coding: utf-8 -*-
"""Тест импортов"""
try:
    import numpy as np
    print("✓ numpy импортирован")
except ImportError as e:
    print(f"✗ Ошибка импорта numpy: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib импортирован")
except ImportError as e:
    print(f"✗ Ошибка импорта matplotlib: {e}")

try:
    from mpl_toolkits.mplot3d import Axes3D
    print("✓ mpl_toolkits.mplot3d импортирован")
except ImportError as e:
    print(f"✗ Ошибка импорта mpl_toolkits.mplot3d: {e}")

print("\nВсе необходимые библиотеки доступны!")
