# -*- coding: utf-8 -*-
import sys
import os

# Добавляем путь к Лаб7
lab7_path = os.path.join(os.path.dirname(__file__), 'Лаб7')
sys.path.insert(0, lab7_path)

# Импортируем и выполняем скрипт
import execute_lab

