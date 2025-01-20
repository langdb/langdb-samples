# Import OpenAI library
from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()

api_key = getenv("LANGDB_API_KEY")

client = OpenAI(
    base_url="https://api.us-east-1.langdb.ai",  # LangDB API base URL,
    api_key=api_key,  # Replace with your LangDB token
)
messages = [
    {
        "role": "user",
        "content": "What is the capital of France?",
    },
]
strong_model = "gpt-4o"
weak_model = "gpt-4o-mini"
routing_body = {
    "extra": {
        "strategy": {"type": "cost", "willingness_to_pay": 0.3},
        "models": [strong_model, weak_model],
    },
}


response = client.chat.completions.create(
    model="router/dynamic",  # Use the router/dynamic model
    messages=messages,  # Define the interaction
    temperature=0.7,  # Control the creativity of the response
    max_tokens=300,  # Limit the length of the response
    top_p=1.0,  # Use nucleus sampling
    extra_headers={"x-project-id": "xxxx"},  # LangDB Project ID
    extra_body=routing_body,
)

print(response.choices[0].message.content)
