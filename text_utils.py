import re

def clean_pdf_text(text: str) -> str:
    """Cleans text extracted from a PDF file."""
    # TODO: Remove References/Bibliography section.
    return remove_citations(text)


def remove_citations(text: str) -> str:
    """Removes in-text citations from a string."""
    # (Author, Year)
    text = re.sub(r'\([A-Za-z0-9,.\s]+\s\d{4}\)', '', text)
    # [1], [2], [3-5], [3, 33, 49, 51]
    text = re.sub(r'\[[0-9,-]+(,\s[0-9,-]+)*\]', '', text)
    return text
