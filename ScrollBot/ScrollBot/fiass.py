

import os
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

def create_vector_store():
    """
    Reads PDFs, extracts text, preprocesses, chunks it, 
    and creates a FAISS vector store for retrieval.
    """

    # ✅ Add all your PDF paths here
    pdf_paths = [
        r"C:\Users\srith\Downloads\rinvoq_pi.pdf",
        r"C:\Users\srith\Downloads\D.pdf"
    ]

    raw_docs = []

    # -----------------------
    # Step 1: Read all PDFs
    # -----------------------
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            continue

        print(f"📄 Reading {pdf_path}...")
        try:
            pdf_reader = PdfReader(pdf_path)

            if len(pdf_reader.pages) == 0:
                print(f"⚠️  No pages found in {pdf_path}")
                continue

            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # Preprocess text → remove excessive whitespace & line breaks
                    cleaned_text = re.sub(r"\s+", " ", text).strip()

                    raw_docs.append(
                        Document(
                            page_content=cleaned_text,
                            metadata={
                                "page": i + 1,
                                "source": os.path.basename(pdf_path)
                            }
                        )
                    )
                else:
                    print(f"⚠️  Empty text on page {i+1} of {pdf_path}")

        except Exception as e:
            print(f"🚨 Error reading {pdf_path}: {str(e)}")

    if not raw_docs:
        print("❌ No document content extracted. PDFs may be scanned images.")
        return

    print(f"✅ Extracted {len(raw_docs)} pages from {len(pdf_paths)} PDF(s)")

    # -----------------------
    # Step 2: Split into chunks
    # -----------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Smaller chunks for better retrieval
        chunk_overlap=100,   # Keeps context between chunks
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(raw_docs)
    print(f"✂️  Split into {len(chunks)} chunks")

    # -----------------------
    # Step 3: Create FAISS Index
    # -----------------------
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_default")


        print("🎉 FAISS index successfully saved at './faiss_index/'")

        # Show a sample for verification
        if chunks:
            print("\n📌 Sample chunk:")
            print("   Source:", chunks[0].metadata)
            print("   Text  :", chunks[0].page_content[:200] + "...")
    except Exception as e:
        print(f"🚨 Error creating/saving FAISS index: {str(e)}")


if __name__ == "__main__":
    create_vector_store()
