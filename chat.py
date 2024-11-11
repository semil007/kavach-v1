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
import requests  # To download image from URL

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

# Dynamically load and set background image from URL
def set_background_image_from_url():
    image_url = "https://skavachbotstorage.blob.core.windows.net/images/sample-11.jpg?sp=r&st=2024-11-11T04:52:41Z&se=2025-11-11T12:52:41Z&spr=https&sv=2022-11-02&sr=b&sig=4WLFvPbpbrNFVbvelAJ49SdCkrlTA1VrfeUEd6hBaiE%3D"
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        st.image(img, use_column_width=True)  # Display as a full-width background

set_background_image_from_url()

# Streamlit app layout
st.title("Kavach Guidelines Chatbot")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# ------------ Authentication Page ------------
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

# ------------ Chatbot Page ------------
def chatbot_page():
    if not st.session_state.logged_in:
        authentication_page()
        return
   
    st.subheader("Welcome to the Chatbot!")

    # Initialize chat history and OCR reader
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    reader = easyocr.Reader(['en'])

    pdf_path = r"D:\final-v1-chatbot\kavach-v1\Annexure-B.pdf"  # Ensure this file exists in the same directory

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

    # Text splitter
    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Chunk creation
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

    # Query input
    if user_query := st.chat_input("Ask a question about Kavach guidelines"):
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Get bot response
        decision = get_kavach_decision(user_query)
        st.session_state.chat_history.append({"role": "assistant", "content": decision})

        # Check for any images if query mentions "image" or "illustration"
        if "image" in user_query.lower() or "illustration" in user_query.lower():
            pdf_document = fitz.open(pdf_path)
            images = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image = Image.open(BytesIO(base_image["image"]))
                    if not contains_text_using_easyocr(image):
                        images.append(image)

            # Display extracted images in a chat-style message if any are found
            if images:
                with st.chat_message("assistant"):
                    st.header("Extracted Kavach-related Images")
                    num_columns = 3
                    cols = st.columns(num_columns)
                    for i, image in enumerate(images):
                        resized_image = image.resize((300, 300), Image.LANCZOS)
                        cols[i % num_columns].image(resized_image, caption=f"Image {i + 1}", use_container_width=True)
            else:
                with st.chat_message("assistant"):
                    st.write("No Kavach-related images found in the document.")

    # Display chat history using `st.chat_message`
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

# Display the appropriate page based on login status
if st.session_state.logged_in:
    chatbot_page()  # Shows the chat interface if logged in
else:
    authentication_page()  # Shows the login page if not logged in
