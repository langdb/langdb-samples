import os
import tempfile
from os import getenv

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants
PROMPT_TEMPLATE = """
You are a banking assistant specializing in answering FAQs about loans, interest rates, and general banking services.
If the user greets, respond with a greeting. If the user asks a question, provide an answer.
Use the following context too for answering questions:

{context}

Conversation History: 
{history}

---


Answer the question based on the above context: {query}

"""

CHROMA_DB_DIR = "chroma"
LANGDB_API_URL = "https://api.us-east-1.langdb.ai/your-project-id/v1"  # Replace with your LANGDB project id
os.environ["LANGDB_API_KEY"] = "your-api-key"

st.set_page_config(page_title="Banking Assistant", layout="wide")
st.title("Banking FAQ Assistant")
st.write("Ask questions about banking services, loan options, and interest rates!")

# Initialize ChromaDB and Embeddings
def initialize_chromadb():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vector_store

# Initialize ChromaDB and LangChain LLM
vector_db = initialize_chromadb()
# Initialize LangChain LLM
llm = ChatOpenAI(
    base_url=LANGDB_API_URL,
    api_key=getenv("LANGDB_API_KEY"),
    model="gpt-3.5-turbo",  # Replace with the specific model name you are using
    timeout=10  # Add a timeout of 10 seconds
)

# Memory for conversation history
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="query",
)

# Prompt Template for LangChain
prompt_template = PromptTemplate(
    input_variables=["context", "history", "query"],
    template=PROMPT_TEMPLATE
)

# LangChain LLM Chain
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

def process_pdf(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getbuffer())

        pdf_loader = PyPDFLoader(temp_file_path)
        documents = pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len)
        chunks = text_splitter.split_documents(documents)

        return chunks

if uploaded_file:
    user_vector_store_dir = CHROMA_DB_DIR
    user_chunks = process_pdf(uploaded_file)
    vector_db.add_documents(user_chunks)
    st.sidebar.success(f"Processed {len(user_chunks)} chunks from uploaded PDF.")

# Chatbox implementation
st.subheader("Chatbox")

# Container for chat messages
chat_container = st.container()

# Function to display chat messages
def display_message(message, is_user=True):
    if is_user:
        chat_container.markdown(f"<div style='text-align: right; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)
    else:
        chat_container.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
with chat_container:
    for chat in st.session_state.messages:
        display_message(chat['content'], is_user=chat['is_user'])

# User Input Section
user_input = st.text_input("Enter your query:", key="user_input")
send_button = st.button("Send")

if send_button:
    user_input = st.session_state.user_input.strip()  # Ensure the input is not empty or just whitespace
    if user_input:
        context = ""
        # Retrieve relevant context from ChromaDB
        try:
            search_results = vector_db.similarity_search(user_input, k=3)
            for result in search_results:
                context += result.page_content + "\n\n"
        except Exception as e:
            st.error(f"Error retrieving context from ChromaDB: {e}")
        try:
            response = chain.run(context=context, query=user_input)
            # Update conversation memory
            st.session_state.messages.append({"role": "user", "content": user_input, "is_user":True})
            st.session_state.messages.append({"role": "assistant", "content": response, "is_user":False})
            st.rerun()
        except requests.exceptions.Timeout:
            st.error("The request to the LLM timed out. Please try again.")
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a valid query.")