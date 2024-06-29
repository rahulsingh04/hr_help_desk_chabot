import os
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Set up Google Generative AI API
google_api_key = "AIzaSyA4-4QffN7x1-Rt5Z7ntrcGE4nxeXia3y4"
google_genai_model = "gemini-pro"
genai.configure(api_key=google_api_key)

# PDF folder path
pdf_folder_path = "Pdfs"

# Function to read all PDF files and return text
def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# Read and process PDF text
text_data = get_pdf_text_from_folder(pdf_folder_path)
text_chunks = get_text_chunks(text_data)

# Set up embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Create FAISS index
faiss = FAISS.from_texts(text_chunks, embeddings)
faiss.save_local("faiss_index")

# Define prompt template
prompt_template = """
    Please provide a detailed answer based on the context provided below. 
    If the information is not available, try to provide the best possible answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Your Answer:
    """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create an instance of ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model=google_genai_model, client=genai, temperature=0.6, google_api_key=google_api_key)

# Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(model, retriever=faiss.as_retriever(), chain_type_kwargs={"prompt": prompt})

def answer_question(question):
    result = qa_chain({"query": question})
    return result["result"]


print(answer_question("hr policy"))

# # Streamlit UI
# st.title("Hyper Hire Chat Bot")

# # Text input with a placeholder
# name = st.text_input("Enter Your Question", placeholder="Type Here:----> ...", key="question_input")

# # Process the question when Enter is pressed
# if name:
#     question = st.session_state.question_input
#     result = answer_question(question)
#     st.success(result)
#     # Clear the text input field
#     st.session_state.question_input = ""
#     st.experimental_rerun()
