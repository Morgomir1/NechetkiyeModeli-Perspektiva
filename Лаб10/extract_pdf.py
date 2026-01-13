# -*- coding: utf-8 -*-
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import PyPDF2
    with open('Лабораторное занятие  10.pdf', 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for i, page in enumerate(reader.pages):
            text += f'\n--- Страница {i+1} ---\n'
            text += page.extract_text() + '\n'
        print(text)
        # Сохраняем в файл для удобства
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("\nТекст сохранен в pdf_content.txt")
except ImportError:
    try:
        import pdfplumber
        with pdfplumber.open('Лабораторное занятие  10.pdf') as pdf:
            text = ''
            for i, page in enumerate(pdf.pages):
                text += f'\n--- Страница {i+1} ---\n'
                text += page.extract_text() + '\n'
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
    sys.exit(1)
