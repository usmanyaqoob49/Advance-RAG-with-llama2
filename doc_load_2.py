from pypdf import PdfReader

def load_doc(path):
    reader = PdfReader(path)
    #reading the pages and removing the whitespaces
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings-->pdf_texts will have only those pages that have the text
    pdf_texts = [text for text in pdf_texts if text]

    return pdf_texts


