import os
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

st.title("Interactive PDF Chatbot with Streamlit")

# Sidebar for chat history
st.sidebar.title("Chat History")
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"
if "history" not in st.session_state:
    st.session_state.history = []

chat_name = st.sidebar.text_input("Chat Name", st.session_state.current_chat)
if st.sidebar.button("Save Chat"):
    if chat_name and chat_name != "New Chat":
        st.session_state.chats[chat_name] = st.session_state.history
        st.session_state.current_chat = chat_name

chat_selection = st.sidebar.radio("Select a Chat", ["New Chat"] + list(st.session_state.chats.keys()))
if chat_selection != st.session_state.current_chat:
    st.session_state.current_chat = chat_selection
    st.session_state.history = st.session_state.chats.get(chat_selection, [])

# Main chat interface
col1, col2 = st.columns([1, 3])

with col2:
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf_file is not None:
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())

        data_chunks = load_pdf("uploaded_pdf.pdf")
        chroma_db = create_embeddings(data_chunks)
        llm = load_model()

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for question, answer in reversed(st.session_state.history):
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Chatbot:** {answer}")
                st.markdown("---")

        # Input box for new questions (fixed at the bottom)
        user_question = st.text_input("Ask a question about the PDF:", key=f"question_input_{len(st.session_state.history)}")

        # Only process the question if it's new (avoid repeating the same question)
        if user_question:
            if user_question.lower() in ['exit', 'quit']:
                st.write("Exiting the program.")
            else:
                answer = predict_ans(chroma_db, llm, user_question)
                st.session_state.history.append((user_question, answer))

                # Refresh the page to show the updated chat
                st.rerun()

    else:
        st.write("Please upload a PDF file to begin.")
