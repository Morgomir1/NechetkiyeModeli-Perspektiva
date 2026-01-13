# -*- coding: utf-8 -*-
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import PyPDF2
    with open('1 занятие R.pdf', 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for i, page in enumerate(reader.pages):
            text += f'\n--- Страница {i+1} ---\n'
            text += page.extract_text() + '\n'
        # Сохраняем в файл для удобства
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Текст извлечен из {len(reader.pages)} страниц и сохранен в pdf_content.txt")
except ImportError:
    try:
        import pdfplumber
        with pdfplumber.open('1 занятие R.pdf') as pdf:
            text = ''
            for i, page in enumerate(pdf.pages):
                text += f'\n--- Страница {i+1} ---\n'
                text += page.extract_text() + '\n'
            with open('pdf_content.txt', 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Текст извлечен из {len(pdf.pages)} страниц и сохранен в pdf_content.txt")
    except ImportError:
        print("Необходимо установить PyPDF2 или pdfplumber")
        print("Попробуйте: pip install PyPDF2 pdfplumber")
        sys.exit(1)
except Exception as e:
    print(f"Ошибка при чтении PDF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
