import os
import PyPDF2
import docx
import email
from email import policy
from email.parser import BytesParser

def extract_pdf_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        for i in range(reader.getNumPages()):
            page = reader.getPage(i)
            text += page.extractText() + "\n"
    return text

def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_email_text(file_path):
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    body = msg.get_body(preferencelist=('plain'))
    if body is not None:
        return body.get_content()
    return ""

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext == ".docx":
        return extract_docx_text(file_path)
    elif ext in [".eml", ".email"]:
        return extract_email_text(file_path)
    else:
        raise ValueError("Unsupported file type: " + ext)