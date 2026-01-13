# -*- coding: utf-8 -*-
"""
Попытка извлечения данных из PDF файла
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pdf_file = 'Лабораторное занятие  10.pdf'

if not os.path.exists(pdf_file):
    print(f"Файл {pdf_file} не найден")
    sys.exit(1)

# Пробуем разные библиотеки
try:
    import PyPDF2
    print("Используется PyPDF2...")
    with open(pdf_file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for i, page in enumerate(reader.pages):
            text += f'\n--- Страница {i+1} ---\n'
            text += page.extract_text() + '\n'
    print("Текст успешно извлечен!")
    print(text)
    with open('pdf_content.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("\nТекст сохранен в pdf_content.txt")
except ImportError:
    try:
        import pdfplumber
        print("Используется pdfplumber...")
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for i, page in enumerate(pdf.pages):
                text += f'\n--- Страница {i+1} ---\n'
                text += page.extract_text() + '\n'
        print("Текст успешно извлечен!")
        print(text)
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("\nТекст сохранен в pdf_content.txt")
    except ImportError:
        print("Необходимо установить PyPDF2 или pdfplumber")
        print("Попробуйте: pip install PyPDF2")
        sys.exit(1)
except Exception as e:
    print(f"Ошибка при чтении PDF: {e}")
    print("\nПожалуйста, откройте PDF файл вручную и введите данные в lab10_calculation.py")
    sys.exit(1)
