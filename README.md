# Interactive PDF Chatbot with Streamlit

This project is an **Interactive PDF Chatbot** built with Streamlit and LangChain. It allows users to upload a PDF, view its contents, and interact with the document by asking questions in natural language. The chatbot utilizes **OpenAI's language models** to provide accurate answers based on the content of the uploaded PDF.

---

## Features

- **PDF Preview:** Displays all pages of the uploaded PDF as images for easy reference.
- **Interactive Chat:** Allows users to ask questions about the content of the PDF and receive answers in real time.
- **Embeddings and Retrieval:** Uses `ChromaDB` and OpenAI embeddings for document vectorization and retrieval.
- **Streamlit Web App:** Easy-to-use interface with a responsive layout.

---

## Requirements

### Python Libraries:
Install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt

1. Clone the Repository:
git clone https://github.com/your-repo/interactive-pdf-chatbot.git
cd interactive-pdf-chatbot

2. Install Dependencies:
pip install -r requirements.txt

3.Set Up Environment Variables:
Create a .env file in the project directory.
OPENAI_API_KEY=your_openai_api_key

4. streamlit run Chat_with_pdf.py


# Usage
1. Upload a PDF:
Upload a PDF file using the file uploader widget in the Streamlit app.

2. View PDF Content:
The app will display all pages of the PDF in the left section.

3. Ask Questions:
Type your question about the PDF in the input box on the right.
The chatbot will process your query and return the most relevant answer.

4.Exit:
Enter "exit" or "quit" in the question box to terminate the program.
