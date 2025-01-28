import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Set Poppler path explicitly (if necessary)
os.environ["POPPLER_PATH"] = r"C:\poppler\bin"  # Replace with your Poppler path

# Set Streamlit page configuration
st.set_page_config(page_title="Interactive PDF Chatbot", layout="wide")

# Load the OpenAI model
def load_model():
    llm = OpenAI()
    return llm

# Load and split the PDF into chunks
def load_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks

# Create embeddings and store them in ChromaDB
def create_embeddings(data):
    embeddings = OpenAIEmbeddings()
    chroma_db = Chroma.from_documents(documents=data, embedding=embeddings, persist_directory='chroma_db')
    chroma_db.persist()  # Persist the database
    return chroma_db

# Predict answer from the question
def predict_ans(chroma_db, llm, question):
    retriever = chroma_db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = chain.run(question)
    return answer

# Function to extract PDF pages and display them
def display_pdf(pdf_path, dpi=300):  # Increase DPI for larger image size
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count
    pdf_images = []
    for i in range(num_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        img_path = f"page_{i}.png"
        # pix.save(img_path)
        pdf_images.append(img_path)
    return pdf_images

# Streamlit app UI
st.title("Interactive PDF Chatbot with Streamlit")

# File uploader in Streamlit
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file is not None:
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())

    data_chunks = load_pdf("uploaded_pdf.pdf")
    chroma_db = create_embeddings(data_chunks)
    llm = load_model()

    # Style for full-screen layout
    st.markdown("""
        <style>
            .pdf-container {
                overflow-y: auto;
                max-height: 700px;
                padding: 10px;
                width: 100%;
            }
            .pdf-image {
                width: 100%;
                margin-bottom: 10px;
            }
            .chat-section {
                margin-top: 20px;
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Full-width columns for PDF and chat
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("PDF Preview")
        pdf_images = display_pdf("uploaded_pdf.pdf")
        for image in pdf_images:
            st.image(image, caption="Page", use_container_width=True)

    with col2:
        st.subheader("Ask a Question")
        user_question = st.text_input("Enter your question about the PDF:")

        if user_question:
            if user_question.lower() in ['exit', 'quit']:
                st.write("Exiting the program.")
            else:
                answer = predict_ans(chroma_db, llm, user_question)
                st.markdown(f"<div class='chat-section'><strong>Answer:</strong> {answer}</div>", unsafe_allow_html=True)
else:
    st.write("Please upload a PDF file to begin.")
