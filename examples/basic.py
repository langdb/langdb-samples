# Import OpenAI library
from openai import OpenAI
api_key = "xxxxx"
client = OpenAI(
    base_url="https://api.us-east-1.langdb.ai",  # LangDB API base URL,
    api_key=api_key,  # Replace with your LangDB token
)
messages = [
    {
        "role": "system",
        "content": "You are a financial assistant. Help the user with their financial queries regarding companies.", 
    },
    {
        "role": "user",
        "content": "What are the earnings of Apple in 2022?",
    },
]
# Make the API call to LangDB's Completions API
response = client.chat.completions.create(
    model="gpt-4o",  # Use the model
    messages=messages,  # Define the interaction
    temperature=0.7,  # Control the creativity of the response
    max_tokens=300,  # Limit the length of the response
    top_p=1.0,  # Use nucleus sampling
    extra_headers={"x-project-id": "xxxxx"} # LangDB Project ID
)