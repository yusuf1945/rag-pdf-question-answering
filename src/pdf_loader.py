# src/pdf_loader.py

from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF file.

    Parameter:
        uploaded_file: PDF file uploaded through Streamlit

    Returns:
        full_text: Extracted text from all pages
    """

    reader = PdfReader(uploaded_file)

    full_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            full_text += f"\n\n--- Page {page_number} ---\n\n"
            full_text += page_text

    return full_text


def clean_text(text):
    """
    Clean extracted text by removing extra spaces and unnecessary line breaks.
    """

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text