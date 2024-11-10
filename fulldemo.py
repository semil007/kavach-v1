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
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel

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

# Background image styling
bg_img = """
<style>
[data-testid="stMain"] {
    background-image: url("https://live.staticflickr.com/65535/53189208896_83b838ad17_k.jpg");
    background-size: cover;
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}
</style>
"""

st.markdown(bg_img, unsafe_allow_html=True)

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

    # Single-click login logic
    if st.button('Login'):
        if authenticate(username, password):
            st.session_state.logged_in = True
        else:
            st.error('Invalid username or password. Please try again.')

# ------------ Chatbot Page ------------
def chatbot_page():
    if not st.session_state.logged_in:
        authentication_page()
        return

    # Show welcome message after login
    st.markdown(
        "<p style='color: white; font-weight: bold;'>Welcome! You are now logged in and can access the chatbot.</p>",
        unsafe_allow_html=True,
    )

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

    # Load CLIP model and processor from transformers
    @st.cache_resource
    def load_clip_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor, device

    model_clip, processor_clip, device_clip = load_clip_model()

    # Image similarity function using CLIP
    def find_relevant_images(query_text, images, threshold=0.22):
        # Compute query embedding
        inputs = processor_clip(text=[query_text], return_tensors="pt", padding=True).to(device_clip)
        with torch.no_grad():
            text_embedding = model_clip.get_text_features(**inputs).cpu().numpy()

        # Compute image embeddings
        image_embeddings = []
        for image in images:
            image_input = processor_clip(images=image, return_tensors="pt").to(device_clip)
            with torch.no_grad():
                image_embedding = model_clip.get_image_features(**image_input).cpu().numpy()
            image_embeddings.append(image_embedding[0])

        # Compute similarities
        similarity_scores = cosine_similarity(text_embedding, image_embeddings)[0]
        relevant_images = [(i, score) for i, score in enumerate(similarity_scores) if score >= threshold]
        return sorted(relevant_images, key=lambda x: x[1], reverse=True)

    # Query input
    if user_query := st.chat_input("Ask a question about Kavach guidelines"):
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Get bot response
        decision = get_kavach_decision(user_query)
        st.session_state.chat_history.append({"role": "assistant", "content": decision})

        # Check for images based on similarity
        if "image" in user_query.lower() or "illustration" in user_query.lower():
            pdf_document = fitz.open(pdf_path)
            images = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    if not contains_text_using_easyocr(image):
                        images.append(image)

            if images:
                relevant_images = find_relevant_images(user_query, images, threshold=0.22)

                if relevant_images:
                    # Display relevant images in a chat-style message
                    with st.chat_message("assistant"):
                        st.write("Relevant Kavach-related Images:")
                        num_columns = 3
                        cols = st.columns(num_columns)
                        for idx, (image_idx, score) in enumerate(relevant_images):
                            resized_image = images[image_idx].resize((250, 250), Image.LANCZOS)
                            caption = f"Similarity: {score:.2f}"
                            cols[idx % num_columns].image(resized_image, caption=caption, use_column_width=True)
                else:
                    with st.chat_message("assistant"):
                        st.write("No relevant images found based on your query.")
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
