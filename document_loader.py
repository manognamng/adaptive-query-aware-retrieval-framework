from PyPDF2 import PdfReader


def load_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Remove References section
    if "References" in text:
        text = text.split("References")[0]

    return text