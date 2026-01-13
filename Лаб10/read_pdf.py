import sys
try:
    import PyPDF2
    with open('Лабораторное занятие  10.pdf', 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        print(text)
except ImportError:
    try:
        import pdfplumber
        with pdfplumber.open('Лабораторное занятие  10.pdf') as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
            print(text)
    except ImportError:
        print("Необходимо установить PyPDF2 или pdfplumber")
        sys.exit(1)
