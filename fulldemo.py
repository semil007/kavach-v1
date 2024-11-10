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
import base64

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

# Authentication function with improved password handling
def authenticate(username, password):
    """Authenticate a user by comparing the hashed password with stored hash."""
    if username in users:
        stored_password_hash = users[username]['password']
        # Hash the provided password to compare with stored hash
        if stored_password_hash == hash_password(password):
            return True
    return False

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Function to encode local image as base64 for HTML embedding
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Paths to local images
bg_img_path = "images/sample-11.jpg"  # Replace with your actual background image path
logo_img_path = "images/logo.jpg.png"  # Replace with your actual logo image path

# Embed CSS and background image
bg_img = f"""
<style>
[data-testid="stMain"] {{
    background-image: url("data:image/jpg;base64,{encode_image(bg_img_path)}");
    background-size: cover;
}}
[data-testid="stHeader"] {{
    background-color: rgba(0, 0, 0, 0);
}}
</style>
"""
st.markdown(bg_img, unsafe_allow_html=True)

# ------------ Authentication Page ------------
def authentication_page():
    st.markdown(f"""
    <link rel="stylesheet" href="styles.css">
    <div class="login-container">
        <h2>Welcome Back</h2>
        <img src="data:image/png;base64,{encode_image(logo_img_path)}" alt="Logo" class="logo">
        <p class="login-subheading">Please enter your credentials to access your account</p>
        <form id="login-form">
          <input id="username" type="text" placeholder="Username" required>
          <input id="password" type="password" placeholder="Password" required>
          <button id="login-button" type="button">Login</button>
        </form>
    </div>
    <script>
    document.getElementById("login-button").onclick = function() {{
        const username = document.getElementById("username").value;
        const password = document.getElementById("password").value;
        if (username === "{users['admin']['email']}" && password === "Kavach2024") {{
            window.location.reload();
        }} else {{
            alert("Invalid username or password. Please try again.");
        }}
    }};
    </script>
    """, unsafe_allow_html=True)

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
    user_query = st.text_input("Enter your query about Kavach guidelines")

    if st.button("Submit Query"):
        if user_query:
            decision = get_kavach_decision(user_query)
            st.session_state.chat_history.append({"user": user_query, "bot": decision})
            st.header("Chat History")
            for chat in st.session_state.chat_history:
                st.write(f"**You**: {chat['user']}")
                st.write(f"**Bot**: {chat['bot']}")

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

                if images:
                    st.header("Extracted Kavach-related Images")
                    num_columns = 3
                    cols = st.columns(num_columns)
                    for i, image in enumerate(images):
                        resized_image = image.resize((300, 300), Image.LANCZOS)
                        cols[i % num_columns].image(resized_image, caption=f"Image {i + 1}", use_container_width=True)
                else:
                    st.write("No Kavach-related images found in the document.")
        else:
            st.warning("Please enter a query.")
    
# Display the appropriate page
if st.session_state.logged_in:
    chatbot_page()
else:
    authentication_page()