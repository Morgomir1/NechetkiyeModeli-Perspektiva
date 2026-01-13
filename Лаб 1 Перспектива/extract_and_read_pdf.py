# -*- coding: utf-8 -*-
"""Извлечение текста из PDF и сохранение для чтения"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pdf_text = ""

try:
    import PyPDF2
    with open('1 занятие R.pdf', 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for i, page in enumerate(reader.pages):
            pdf_text += f'\n--- Страница {i+1} ---\n'
            pdf_text += page.extract_text() + '\n'
    print("Использован PyPDF2")
except ImportError:
    try:
        import pdfplumber
        with pdfplumber.open('1 занятие R.pdf') as pdf:
            for i, page in enumerate(pdf.pages):
                pdf_text += f'\n--- Страница {i+1} ---\n'
                pdf_text += page.extract_text() + '\n'
        print("Использован pdfplumber")
    except ImportError:
        print("ОШИБКА: Необходимо установить PyPDF2 или pdfplumber")
        print("pip install PyPDF2 pdfplumber")
        sys.exit(1)
except Exception as e:
    print(f"ОШИБКА при чтении PDF: {e}")
    sys.exit(1)

# Сохраняем текст
with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write(pdf_text)

print(f"Текст сохранен в pdf_content.txt (длина: {len(pdf_text)} символов)")

# Выводим первые 5000 символов для проверки
print("\n" + "="*60)
print("ПЕРВЫЕ 5000 СИМВОЛОВ:")
print("="*60)
print(pdf_text[:5000])
