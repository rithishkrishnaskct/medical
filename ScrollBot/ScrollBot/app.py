from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from markupsafe import Markup

# -------------------------------
# Load environment + API
# -------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = "your_secret_key"  # replace with strong key
users = {}  # Temporary user store

# -------------------------------
# Text Cleaning
# -------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'covid[-\s]?19', 'covid19', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# Conversational Chain with History
# -------------------------------
def get_conversational_chain():
    prompt_template = """
    You are MediBot, a medical Q&A assistant. 
    Use the provided context and also consider the previous conversation. 

    Conversation so far:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    If the answer is not in the context, reply:
    "answer is not available in the context".

    Answer in a clear and structured way.
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# -------------------------------
# FAISS + History Response
# -------------------------------
def get_response(user_question, chat_history):
    question = clean_text(user_question)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Pick which FAISS index to use: uploaded or default
    index_path = session.get("knowledge_base", "faiss_index_default")

    try:
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return {"answer": "No knowledge base found. Please upload/build FAISS index first.", "sources": []}

    docs = new_db.similarity_search(question, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Join history into string
    history_text = ""
    for turn in chat_history:
        if "question" in turn:
            history_text += f"User: {turn['question']}\n"
        history_text += f"Bot: {Markup(turn['answer']).striptags()}\n"

    chain = get_conversational_chain()
    response = chain.run(chat_history=history_text, context=context, question=question)

    if "answer is not available in the context" in response.lower():
        return {"answer": response, "sources": []}
    
    sources = []
    for doc in docs:
        metadata = doc.metadata
        page = metadata.get("page", "N/A")
        source = metadata.get("source", "Uploaded PDF")
        sources.append(f"{source} (Page {page})")


    return {"answer": response, "sources": sources}

# -------------------------------
# Format Answer for UI
# -------------------------------
def format_answer(answer_dict):
    text = answer_dict["answer"]
    sources = answer_dict["sources"]

    formatted = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("*"):
            formatted += f"<li>{line[1:].strip()}</li>"
        elif line:
            formatted += f"<p>{line}</p>"
    if "<li>" in formatted:
        formatted = "<ul>" + formatted + "</ul>"

    if sources:
        source_html = "<div style='margin-top: 8px;'>"
        source_html += "<strong>Sources:</strong><br>" + "<br>".join(sources) + "</div>"
        formatted += source_html

    return Markup(formatted)

# -------------------------------
# Flask Routes
# -------------------------------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname] == pwd:
            session['user'] = uname
            session['knowledge_base'] = "faiss_index_default"
            return redirect(url_for('chatbot'))
        else:
            error = "Invalid username or password"
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users:
            return "User already exists"
        users[uname] = pwd
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'chat_history' not in session:
        session['chat_history'] = []
        session['chat_history'].append({
            'answer': Markup("<p>Hi, I am <b>MediBot</b>. How can I assist you today?</p>")
        })
        session.modified = True

    if request.method == 'POST':
        question = request.form['question']
        q_lower = question.lower().strip()

        # Handle greetings/small talk without hitting PDFs
        smalltalk = {
            "hi": "Hi! I’m MediBot. How can I assist you today?",
            "hello": "Hello! How can I help you today?",
            "hey": "Medibot there! What would you like to know?",
            "good morning": "Good morning! How can I assist you today?",
            "good evening": "Good evening! How can I help you?"
        }

        if q_lower in smalltalk:
            answer = Markup(f"<p>{smalltalk[q_lower]}</p>")
        else:
            raw_answer = get_response(question, session['chat_history'])
            answer = format_answer(raw_answer)

        session['chat_history'].append({'question': question, 'answer': answer})
        session.modified = True

    return render_template('chatbot.html', chat_history=session.get('chat_history', []))

@app.route('/upload', methods=['POST'])
def upload():
    # ✅ FIX: match input name from chatbot.html
    if 'file' not in request.files:
        return redirect(url_for('chatbot'))

    pdf = request.files['file']
    if pdf.filename == '':
        return redirect(url_for('chatbot'))

    save_path = os.path.join("uploads", pdf.filename)
    os.makedirs("uploads", exist_ok=True)
    pdf.save(save_path)

    loader = PyPDFLoader(save_path)
    pages = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(pages, embeddings)

    # ✅ unique FAISS index per user
    user_index = os.path.join("faiss_index_" + session['user'])
    vectorstore.save_local(user_index)

    session['knowledge_base'] = user_index
    return redirect(url_for('chatbot'))


@app.route('/clear_chat')
def clear_chat():
    session['chat_history'] = []
    session['chat_history'].append({
        'answer': Markup("<p>Hi, I am <b>MediBot</b>. How can I assist you today?</p>")
    })
    session.modified = True
    return redirect(url_for('chatbot'))

@app.route('/reset_kb')
def reset_kb():
    # ✅ frontend fetch expects JSON
    session['knowledge_base'] = "faiss_index_default"
    return {"message": "Knowledge base reset to default."}

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('knowledge_base', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
