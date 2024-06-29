import streamlit as st
from PIL import Image
from gtts import gTTS
import tempfile
import os
from playsound import playsound
import base64

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

load_dotenv()

# Configuration
google_api_key = "AIzaSyA4-4QffN7x1-Rt5Z7ntrcGE4nxeXia3y4"
google_genai_model = "gemini-pro"
pdf_folder_path = "Pdfs"

# Initialize embeddings and vector database
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key) 
vector_database = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 



def get_response(question):
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, please think rationally and answer from your own knowledge base 
    {context}
    Question: {question}
    """
    model = ChatGoogleGenerativeAI(model=google_genai_model, client=genai, temperature=0.3, 
                                   google_api_key=google_api_key,
                                   transport='rest')
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "chat_history", "question"])
    
    combine_custom_prompt = '''
    Generate a summary of the following text that includes the following elements:
    * A title that accurately reflects the content of the text.
    * An introduction paragraph that provides a solution or answer to the topic.
    * Bullet points that list the key points that contain the solution or answer.
    * A conclusion paragraph that summarizes the main points of the text.
    Text:`{context}`
    '''
    combine_prompt_template = PromptTemplate(template=combine_custom_prompt, 
                                             input_variables=['context'])
    
    retriever = vector_database.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25})

    chain_type_kwargs = {
        "verbose": True,
        "question_prompt": prompt,
        "combine_prompt": combine_prompt_template,
        "combine_document_variable_name": "context",
        "memory": ConversationBufferMemory(
            llm=model,
            memory_key="chat_history",
            input_key="question",
            return_messages=False)
    }
    
    qa_chain = RetrievalQA.from_chain_type(model, retriever=retriever, 
                                           chain_type='map_reduce',
                                           chain_type_kwargs=chain_type_kwargs)
    
    result = qa_chain.invoke({"query": question})
    return result['result']




# Display image
st.title('🦜:blue[H-R Help Desk] :sunglasses:')
hr_image = Image.open(r"image_dir/images.jpeg")
st.image(hr_image, width=300)


# Input field
name = st.text_input("Enter Your Question", placeholder="Please Write Answer Related To H-R Policy", key="question_input")

if name:
    question = st.session_state.question_input
    result = get_response(question)
    
    # Display the result in a text area
    st.text_area("See Full Result ", result, height=100)
    
    # Optionally, use an expander to display the result
    with st.expander("RESULT"):
        st.write(result)
    
