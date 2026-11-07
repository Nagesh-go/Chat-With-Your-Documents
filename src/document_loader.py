import os
import sys
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
def load_single_pdf(file_path: str) -> List[Document]:
    """
    Loads a single PDF file and returns a list of LangChain Document objects.
    Each page in the PDF becomes a single Document object.
    
    Args:
        file_path (str): The relative or absolute path to the PDF file.
        
    Returns:
        List[Document]: A list of Document objects with page_content and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {os.path.abspath(file_path)}")
    
    print(f"[INFO] Loading PDF from: {file_path}")
    
    # Initialize the PyPDFLoader
    loader = PyPDFLoader(file_path=file_path)
    
    # Load and extract pages
    documents = loader.load()
    
    print(f"[SUCCESS] Successfully loaded {len(documents)} page(s) from '{os.path.basename(file_path)}'.")
    return documents
def inspect_documents(documents: List[Document], preview_chars: int = 250):
    """
    Prints diagnostic metadata and a content preview for each loaded page.
    """
    print("\n" + "=" * 65)
    print("DOCUMENT INSPECTION & METADATA OVERVIEW")
    print("=" * 65)
    
    for idx, doc in enumerate(documents):
        page_num = doc.metadata.get("page", 0)  # Note: PyPDFLoader uses 0-based page indexing
        source_path = doc.metadata.get("source", "Unknown")
        content_length = len(doc.page_content)
        
        print(f"\n--- Page Object Index [{idx}] (Physical Page {page_num + 1}) ---")
        print(f"Metadata : {doc.metadata}")
        print(f"Length   : {content_length} characters")
        print(f"Preview  :\n{'-'*40}")
        # Clean up whitespace for preview
        preview_text = doc.page_content.strip()[:preview_chars]
        print(preview_text if preview_text else "[EMPTY PAGE OR SCANNED IMAGE]")
        print(f"{'-'*40}")
if __name__ == "__main__":
    # Path to our target document
    sample_pdf_path = os.path.join("data", "sample.pdf")
    
    # Fallback check: If sample.pdf doesn't exist, search for any .pdf in data/
    if not os.path.exists(sample_pdf_path):
        data_dir = "data"
        found_pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")] if os.path.exists(data_dir) else []
        if found_pdfs:
            sample_pdf_path = os.path.join(data_dir, found_pdfs[0])
        else:
            print(f"[ERROR] No PDF files found in '{data_dir}/'.")
            print("Please place a PDF file in the 'data/' folder and name it 'sample.pdf'.")
            sys.exit(1)
            
    # Execute the load and inspection
    docs = load_single_pdf(sample_pdf_path)
    inspect_documents(docs)