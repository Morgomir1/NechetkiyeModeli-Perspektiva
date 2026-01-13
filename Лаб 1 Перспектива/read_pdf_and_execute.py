# -*- coding: utf-8 -*-
"""
Скрипт для извлечения текста из PDF и понимания задания
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Попытка извлечь текст из PDF
pdf_text = ""
try:
    import PyPDF2
    with open('1 занятие R.pdf', 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            pdf_text += f'\n{"="*60}\nСТРАНИЦА {i+1}\n{"="*60}\n'
            pdf_text += page.extract_text() + '\n'
    print(f"Извлечено {len(reader.pages)} страниц")
except ImportError:
    try:
        import pdfplumber
        with pdfplumber.open('1 занятие R.pdf') as pdf:
            for i, page in enumerate(pdf.pages):
                pdf_text += f'\n{"="*60}\nСТРАНИЦА {i+1}\n{"="*60}\n'
                pdf_text += page.extract_text() + '\n'
        print(f"Извлечено {len(pdf.pages)} страниц")
    except ImportError:
        print("ОШИБКА: Установите PyPDF2 или pdfplumber: pip install PyPDF2 pdfplumber")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА (pdfplumber): {e}")
        sys.exit(1)
except Exception as e:
    print(f"ОШИБКА (PyPDF2): {e}")
    sys.exit(1)

# Сохраняем текст
with open('pdf_full_text.txt', 'w', encoding='utf-8') as f:
    f.write(pdf_text)

print(f"\nТекст сохранен в pdf_full_text.txt")
print(f"Длина: {len(pdf_text)} символов\n")

# Выводим первые 3000 символов для анализа
print("="*60)
print("НАЧАЛО ТЕКСТА ИЗ PDF (первые 3000 символов):")
print("="*60)
print(pdf_text[:3000])
print("\n" + "="*60)
