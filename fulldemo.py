import streamlit as st
import hashlib
import os
from PyPDF2 import PdfReader
from PIL import Image
from io import BytesIO
import fitz  # PyMuPDF for extracting images from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import easyocr  # For OCR-based text extraction
import numpy as np  # For array conversion

# Load environment variables from .env file (for local development)
load_dotenv()

# Helper function to hash passwords for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Predefined credentials with hashed passwords
users = {
    'admin': {
        'email': 'admin@example.com',
        'name': 'Admin User',
        'password': hash_password('Kavach2024'),
    }
}

# Authentication function
def authenticate(username, password):
    if username in users and users[username]['password'] == hash_password(password):
        return True
    return False

# Streamlit app layout
st.title("Kavach Guidelines Chatbot")

# Background and style
bg_img = """
<style>
[data-testid="stMain"] {
background-image: url("https://www.railway-technology.com/wp-content/uploads/sites/13/2018/06/indianrailways.jpg");
background-size: cover;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}

.chat-container {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.9);
    margin-top: 20px;
}

.chat-message {
    background-color: white;
    padding: 8px;
    border-radius: 5px;
    margin-bottom: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.button-container {
    display: flex;
    justify-content: space-between;
}
</style>
"""
st.markdown(bg_img, unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Authentication Page
def authentication_page():
    st.subheader("Login to Access Chatbot")
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.success(f"Welcome {users[username]['name']}! Redirecting to chatbot...")
        else:
            st.error('Invalid username or password. Please try again.')

# Chatbot Page
def chatbot_page():
    if not st.session_state.logged_in:
        authentication_page()
        return
    
    st.subheader("Welcome to the Chatbot!")
    
    # Initialize OCR reader and load PDF
    reader = easyocr.Reader(['en'])
    pdf_path = "Annexure-B.pdf"  # Ensure this file exists in the same directory

    # OCR helper function
    def contains_text_using_easyocr(image):
        image_np = np.array(image)
        result = reader.readtext(image_np)
        return len(result) > 0

    # Load and process PDF
    @st.cache_resource
    def load_pdf():
        pdf_reader = PdfReader(pdf_path)
        kavach_text = ''.join(page.extract_text() or "" for page in pdf_reader.pages)
        return kavach_text

    kavach_text = load_pdf()

    # Text splitter and chunk creation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    @st.cache_resource
    def create_chunks(kavach_text):
        return [{'text': chunk, 'source': 'kavach_source'} for chunk in text_splitter.split_text(kavach_text)]
    kavach_chunks = create_chunks(kavach_text)

    # FAISS Vector Store
    @st.cache_resource
    def create_vector_store(kavach_chunks):
        embeddings = HuggingFaceEmbeddings()
        return FAISS.from_texts(
            texts=[chunk['text'] for chunk in kavach_chunks],
            embedding=embeddings,
            metadatas=[{'source': chunk['source']} for chunk in kavach_chunks]
        )
    vectorstore = create_vector_store(kavach_chunks)

    # Configure Google Gemini Pro API
    api_key = os.getenv("MY_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("API Key not found. Ensure 'MY_API_KEY' is set in environment variables.")
        return

    # Query handling functions
    def generate_kavach_response(prompt):
        model = genai.GenerativeModel('gemini-1.5-pro-002')
        response = model.generate_content(prompt)
        return response.text

    def get_kavach_decision(query):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(query)
        combined_text = "\n".join([doc.page_content for doc in relevant_docs])
        
        if not combined_text.strip():
            return "No relevant information found based on your query."
        
        prompt = f"Based on the following Kavach guidelines:\n\n{combined_text}\n\nAnswer the query: {query}"
        return generate_kavach_response(prompt)

    # Input and chat display
    user_query = st.text_input("Enter your query about Kavach guidelines")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("Submit Query")
    with col2:
        clear_chat = st.button("Clear Chat History")

    if submit and user_query:
        decision = get_kavach_decision(user_query)
        st.session_state.chat_history.append({"user": user_query, "bot": decision})

    if clear_chat:
        st.session_state.chat_history = []

    # Display chat history in a scrollable container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.write(f"**You**: {chat['user']}")
        st.markdown(f"<div class='chat-message'><strong>Bot</strong>: {chat['bot']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Display the appropriate page
if st.session_state.logged_in:
    chatbot_page()
else:
    authentication_page()
