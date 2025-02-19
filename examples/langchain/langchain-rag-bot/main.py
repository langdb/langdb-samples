import os
from os import getenv

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import requests
from langchain_openai import ChatOpenAI

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
LANGDB_API_URL = "https://api.us-east-1.langdb.ai/your-project-d/v1"  # Replace with your LANGDB project id
os.environ["LANGDB_API_KEY"] = "your-api-key"

st.set_page_config(page_title="Banking Assistant", layout="wide")
st.title("Banking FAQ Assistant")
st.write("Ask questions about banking services, loan options, and interest rates!")

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
        try:
            context = ""
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