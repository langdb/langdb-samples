import os

from llama_index.llms import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_BASE"] = "https://api.us-east-1.langdb.ai"

Settings.llm = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)


documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist('storage')
query_engine = index.as_query_engine()
response = query_engine.query("what are features of langdb?")
print(response)






