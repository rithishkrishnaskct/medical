# 🩺 MediBot — Intelligent Medical PDF Chatbot Using RAG + Google GenAI + Flask + FAISS

MediBot is an AI-powered medical assistant that allows users to upload **drug information PDFs**—such as prescribing information, patient queries, medical guidelines, and research documents—and ask natural-language questions to retrieve accurate, citation-supported answers.

Built using **Retrieval-Augmented Generation (RAG)**, MediBot turns static medical PDFs into an interactive, intelligent, and helpful chatbot designed for medical-students, clinicians, pharmacists, and researchers.

---

## ⭐ Key Features

### 📄 PDF Upload + Extraction
- Upload any medical/healthcare PDF.
- Automatic text extraction with **page number tracking**.
- Extracts clean, structured text ready for semantic search.

### 🧬 Embeddings using Google GenAI
- Converts text chunks into semantic vectors.
- Ensures accurate medical context understanding.

### ⚡ High-Speed Search using FAISS
- FAISS vector index enables **millisecond-level** search.
- Retrieves the most relevant medical chunks for each query.

### 🤖 GenAI-Powered Answers
- Google Gemini creates final responses.
- Answers are grounded in retrieved PDF content.

### 📚 Citations Included
Each answer contains:
- PDF name  
- Page number  
- Referenced medical snippet  

### 🌐 Flask Web Application
Includes:
- User login  
- User registration  
- PDF upload  
- Chat interface  
- Session handling
  
# 🗂️ Complete Project Structure

Below is the full MediBot directory structure:
```bash
MediBot/
│
├── app.py # Main Flask app: routes, backend logic, chat engine
├── fiass.py # PDF processing, embeddings, FAISS index creation
├── test.py # Debug/testing purposes
│
├── requirements.txt # Python dependencies
├── .env # Contains GOOGLE_API_KEY (created by user)
├── README.md # Documentation (this file)
│
├── uploads/ # Folder for user-uploaded medical PDFs
│
├── faiss_index/ # Vector storage for semantic search
│ ├── index.faiss # FAISS index file
│ └── index.pkl # Metadata file (mapping chunks → text/pages)
│
├── templates/ # Frontend HTML templates (Flask)
│ ├── login.html
│ ├── register.html
│ └── chatbot.html
│
└── images/ # Optional UI screenshots for README (not required)
```
# 🧠 How MediBot Works (RAG Pipeline)

### 1️⃣ PDF Upload  
User uploads a medical PDF → stored in the `/uploads` folder

### 2️⃣ PDF Processing  
`fiass.py` handles:
- Text extraction  
- Chunking with overlap  
- Page number tracking  

### 3️⃣ Embedding Generation  
Each chunk → converted to an embedding using **Google GenAI embeddings**.

### 4️⃣ FAISS Vector Index  
Embeddings stored in FAISS:
- Enables fast semantic search  
- Helps retrieve correct medical information  

### 5️⃣ User Query  
User types a medical question.

### 6️⃣ Retrieval + LLM  
- Relevant chunks retrieved from FAISS  
- Sent to Gemini  
- Gemini generates a **grounded, accurate medical answer**

### 7️⃣ Response Returned  
Chat UI displays:
- Final answer  
- Citations  
- Page references  
- Source PDF
  
# 🌐 Web Interface (Flask Frontend)

### 🔐 Login & Register (Authentication)
Secure user access using:
- `login.html`
- `register.html`

### 💬 Chat Interface
Modern UI for:
- Chatting with MediBot  
- Uploading PDFs  
- Clearing chat  
- Logging out  

# ▶️ How to Run MediBot (Step-by-Step)
```bash
1. Clone the Repository
git clone https://github.com/your-username/MediBot.git
cd MediBot

2. Create a Virtual Environment
python -m venv .venv
Activate Environment
Windows:
.venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Create .env File and Add API Key
Inside the root folder, create a .env file:
GOOGLE_API_KEY=your_google_genai_api_key_here

5. Run the Flask App
python app.py

6. Open in Browser
http://localhost:5000
```
