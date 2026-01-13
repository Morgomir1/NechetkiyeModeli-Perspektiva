# -*- coding: utf-8 -*-
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
    with open('pdf_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"OK: {len(reader.pages)} страниц")
except ImportError:
    try:
        import pdfplumber
        with pdfplumber.open('1 занятие R.pdf') as pdf:
            text = ''
            for i, page in enumerate(pdf.pages):
                text += f'\n--- Страница {i+1} ---\n'
                text += page.extract_text() + '\n'
        with open('pdf_text.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"OK: {len(pdf.pages)} страниц")
    except ImportError:
        print("ERROR: Установите PyPDF2 или pdfplumber")
    except Exception as e:
        print(f"ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")
