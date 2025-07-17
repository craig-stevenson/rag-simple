from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
import warnings
from typing import List
import sys
import os
import pickle

def get_pages_as_list(pdf_path)->List[str]:
    pages = []
    with warnings.catch_warnings():
        reader = PdfReader(pdf_path, strict=False)
        print(f"num_pages={len(reader.pages)}")
        
        for i, page in enumerate(reader.pages):
            try:
                if i%50 == 0:
                    size_bytes = sys.getsizeof(pages)
                    size_kb = size_bytes / 1024
                    print(f"processing page {i}")
                    
                text = page.extract_text()
                if text.strip():
                    pages.append(text)
            except Exception as e:
                print(f"skipping page {i} due to error: {e}")
   
    return pages

def pdf_to_markdown(pdf_path):
    pages: List[str] = get_pages_as_list(pdf_path)
    full_text = ''.join(pages)
    
    # Generate markdown filename: replace .pdf extension with .md
    base_name = os.path.splitext(pdf_path)[0]
    markdown_path = f"{base_name}.md"
    
    # Write full_text to markdown file
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Markdown file created: {markdown_path}")
    return markdown_path
        
def pdf_to_document_list(file_path)->list[Document]:
    with warnings.catch_warnings():
        loader = PyPDFLoader(file_path)
        pages = []
        for i,page in enumerate(loader.lazy_load()):
            try:
                if i%50 == 0:
                    print(f"processing page {i}")
                pages.append(page)
            except Exception as e:
                print(f"skipping page {i} due to error: {e}")
        return pages

def pdf_to_str(file_path)->str:
    with warnings.catch_warnings():
        loader = PyPDFLoader(file_path)
        pages = []
        for i,page in enumerate(loader.lazy_load()):
            try:
                if i%50 == 0:
                    print(f"processing page {i}")
                pages.append(page.page_content)
            except Exception as e:
                print(f"skipping page {i} due to error: {e}")
        return ''.join(pages)
    
def pdf_to_chunks(file_path):
    doc_list: list[Document] = pdf_to_document_list(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_documents(doc_list)
    return chunks

def docs_to_chunks(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)
    return chunks

def add_id_to_chunks(chunks):
    pass

def create_chunks(file_path: str):
    if file_path.endswith(".pdf"):
        return pdf_to_chunks(file_path)
    return []

def read_chunks(file_path: str):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
        return loaded_data

if __name__ == "__main__":
    # Test the pdf_to_markdown function
    test_pdf_path = "insert_pdf_path_here"
    
    docs = pdf_to_document_list(test_pdf_path)
    chunks = docs_to_chunks(docs)
    
        
    print(f"docs={len(docs)}, chunks={len(chunks)}")
    
