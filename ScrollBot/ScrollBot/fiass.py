# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import os
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from dotenv import load_dotenv

# # load_dotenv()
# # os.getenv("GOOGLE_API_KEY")
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# # def get_text_chunks(text):
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# # def get_vector_store(text_chunks):
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
# #     vector_store.save_local("faiss_index")

# # raw_text = get_pdf_text([r"C:\Users\srith\Downloads\rinvoq_pi.pdf"])
# # text_chunks = get_text_chunks(raw_text)
# # get_vector_store(text_chunks)


# # pdf_processor.py
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def create_vector_store():
#     pdf_path = r"C:\Users\srith\Downloads\rinvoq_pi.pdf"  # ← Update if needed

#     if not os.path.exists(pdf_path):
#         print(f"❌ PDF not found: {pdf_path}")
#         return

#     print("📄 Reading PDF...")
#     pdf_reader = PdfReader(pdf_path)
#     raw_docs = []

#     for i, page in enumerate(pdf_reader.pages):
#         text = page.extract_text()
#         if text and text.strip():
#             raw_docs.append(
#                 Document(
#                     page_content=text,
#                     metadata={
#                         "page": i + 1,
#                         "source": os.path.basename(pdf_path)
#                     }
#                 )
#             )

#     print(f"✅ Extracted {len(raw_docs)} pages with metadata")

#     # Split
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_documents(raw_docs)
#     print(f"✂️  Split into {len(chunks)} chunks")

#     # Embed and save
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

#     print("🎉 FAISS index saved with metadata!")
#     print("📌 Sample:", chunks[0].metadata, " | Preview:", chunks[0].page_content[:100])

# if __name__ == "__main__":
#     create_vector_store()



# # pdf_processor.py
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# def create_vector_store():
#     # ✅ Add all your PDFs here
#     pdf_paths = [
#         r"C:\Users\srith\Downloads\rinvoq_pi.pdf"
#     ]

#     raw_docs = []

#     for pdf_path in pdf_paths:
#         if not os.path.exists(pdf_path):
#             print(f"❌ PDF not found: {pdf_path}")
#             continue

#         print(f"📄 Reading {pdf_path}...")
#         try:
#             pdf_reader = PdfReader(pdf_path)
#             if len(pdf_reader.pages) == 0:
#                 print(f"⚠️  No pages found in {pdf_path}")
#                 continue

#             for i, page in enumerate(pdf_reader.pages):
#                 text = page.extract_text()
#                 if text and text.strip():
#                     # Clean up strange characters or line breaks
#                     text = re.sub(r'\s+', ' ', text).strip()
#                     raw_docs.append(
#                         Document(
#                             page_content=text,
#                             metadata={
#                                 "page": i + 1,
#                                 "source": os.path.basename(pdf_path)
#                             }
#                         )
#                     )
#                 else:
#                     print(f"⚠️  Empty text on page {i+1} of {pdf_path}")
#         except Exception as e:
#             print(f"🚨 Error reading {pdf_path}: {str(e)}")

#     if not raw_docs:
#         print("❌ No document content extracted. Check if PDFs are scanned images.")
#         return

#     print(f"✅ Extracted {len(raw_docs)} pages from {len(pdf_paths)} PDFs")

#     # Split into meaningful chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,           # Slightly smaller for better retrieval
#         chunk_overlap=100,        # Overlap preserves context
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     chunks = text_splitter.split_documents(raw_docs)
#     print(f"✂️  Split into {len(chunks)} chunks")

#     # Use consistent embedding model
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # Create and save FAISS index
#     try:
#         vector_store = FAISS.from_documents(chunks, embedding=embeddings)
#         vector_store.save_local("faiss_index")
#         print("🎉 FAISS index saved at './faiss_index/'")
        
#         # Print sample
#         if chunks:
#             print("📌 Sample chunk:")
#             print("   Source:", chunks[0].metadata)
#             print("   Text:", chunks[0].page_content[:150] + "...")
#     except Exception as e:
#         print("🚨 Error saving FAISS index:", str(e))


# if __name__ == "__main__":
#     import re  # Added missing import
#     create_vector_store()




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
        r"C:\Users\srith\Downloads\rinvoq_pi.pdf"
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
        vector_store.save_local("faiss_index")

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
