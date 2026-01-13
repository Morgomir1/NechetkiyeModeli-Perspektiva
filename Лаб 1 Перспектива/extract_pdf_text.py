# -*- coding: utf-8 -*-
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

text_content = ""

# Попытка 1: PyPDF2
try:
    import PyPDF2
    with open('1 занятие R.pdf', 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text_content += f'\n=== Страница {i+1} ===\n'
            text_content += page.extract_text() + '\n'
    print(f"Успешно извлечено из {len(reader.pages)} страниц с помощью PyPDF2")
except ImportError:
    # Попытка 2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open('1 занятие R.pdf') as pdf:
            for i, page in enumerate(pdf.pages):
                text_content += f'\n=== Страница {i+1} ===\n'
                text_content += page.extract_text() + '\n'
        print(f"Успешно извлечено из {len(pdf.pages)} страниц с помощью pdfplumber")
    except ImportError:
        print("ОШИБКА: Необходимо установить PyPDF2 или pdfplumber")
        print("Выполните: pip install PyPDF2 pdfplumber")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА при чтении PDF (pdfplumber): {e}")
        sys.exit(1)
except Exception as e:
    print(f"ОШИБКА при чтении PDF (PyPDF2): {e}")
    sys.exit(1)

# Сохранение текста
output_file = 'pdf_extracted_text.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(text_content)

print(f"\nТекст сохранен в файл: {output_file}")
print(f"Длина текста: {len(text_content)} символов")
print("\nПервые 2000 символов:")
print("=" * 60)
print(text_content[:2000])
