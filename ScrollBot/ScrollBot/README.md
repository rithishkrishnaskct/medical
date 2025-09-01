🚀 ScrollBot — An Intelligent PDF Chatbot Using RAG with Google GenAI, Flask, and FAISS
ScrollBot is an AI-powered assistant that lets students interact with academic PDFs — like syllabus documents, past year question papers, and more — through natural language questions.

Built using a Retrieval-Augmented Generation (RAG) approach, it transforms traditional, non-interactive PDFs into a smart, conversational experience. Whether you're preparing for exams or exploring your course structure, ScrollBot makes it easy to access the information you need — fast and effortlessly.

🗂️ Project Structure
ScrollBot/
│
├── faiss_index/         # Stores FAISS vector database
│   ├── index.faiss
│   └── index.pkl
│
├── templates/           # Flask HTML templates
│   ├── chatbot.html
│   ├── login.html
│   └── register.html
│
├── images/              # UI/branding screenshots (for README or UI assets)
│
├── .env                 # Environment variables (e.g., API keys)
├── app.py               # Main Flask application (routing + frontend logic)
├── fiass.py             # FAISS + PDF processing logic (backend intelligence)
├── test.py              # For debugging or testing routes
├── requirements.txt     # Python dependencies
└── README.md            # You're reading it!

🧠 How It Works
This RAG-based system follows the steps below:

1.PDF Upload & Text Chunking
Documents are loaded and split into manageable chunks.

2.Semantic Vectorization
Each chunk is embedded using Google GenAI Embeddings.

3.Indexing via FAISS
These vectors are stored in a FAISS vector database for fast retrieval.

4.User Interaction
User queries are processed, and the most relevant document content is returned as a chat response.

🌐 Web Interface (Frontend)
🔐 Login/Register
The app starts with a clean and simple user authentication flow:
templates/
├── login.html
├── register.html
└── chatbot.html
💬 Chat Interface
Once logged in, users can interact with ScrollBot through a smooth, modern chat interface featuring:

Welcome messages

Logout and clear chat options

Clean UI styled with HTML/CSS

🔧 Behind the Scenes (Backend)
fiass.py: Vector creation & PDF handling logic

app.py: Flask app with routes, sessions, and chat logic

.env: Secure storage for API keys

index.faiss & index.pkl: Preprocessed FAISS data


▶️ How to Run ScrollBot

1.Clone the repository
git clone https://github.com/your-username/ScrollBot.git

2.Change into the project directory
cd ScrollBot

3.(Optional but recommended) Create a virtual environment
python -m venv .venv

4.Activate the virtual environment
On Windows:
.venv\Scripts\activate

On macOS/Linux:
source .venv/bin/activate

5.Install the required Python packages
pip install -r requirements.txt

6.Create a .env file in the root directory and add your API key
GOOGLE_API_KEY=your_google_genai_api_key_here

7.Run the Flask app
python app.py

8.Open your browser and go to
http://localhost:5000
