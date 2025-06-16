import requests
import json
from openai import OpenAI
from uuid import uuid4

api_key = "langdb_NjVXcFI0SjZidmROcCtGa0ZKSVVVMVVYMnMwY2JjUGVmVy93elJjNEpOT3FNK0lQbWJ2aUZqWEZSb2M2TFZxQmozWUpOcy9kR0JIWS9TbmtOMktUVkdXem82bXJCOGMxeVJuVUdjazdFekh0QnpxZElTaFZOaVZJOHFhUTZrVGgzUUJXRlZTaUFqUS9Uc3pkY2Q2Y3BraXUwQVJ1WlcvNERWeWhMcTJ0KzVZZ0wzTHlyYXF4cW1UakFpc1ZTV05LRTVIREd3PT06QUFBQUFBQUFBQUFBQUFBQQ=="
url = "https://api.staging.langdb.ai/v1/chat/completions"
import logging
from datetime import datetime


def setup_logging():
    """Set up logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"model_test_errors_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    return log_filename


log_filename = setup_logging()


def log_request(method, url, **kwargs):
    """Log request details before making the request"""
    logging.info(f"Request: {method.upper()} {url}")
    logging.info(f"Request Headers: {kwargs.get('headers', {})}")
    if "json" in kwargs:
        logging.info(f"Request Body: {json.dumps(kwargs['json'], indent=2)}")
    elif "data" in kwargs:
        logging.info(f"Request Data: {kwargs['data']}")

    response = requests.request(method, url, **kwargs)

    logging.info(f"Response Status: {response.status_code}")
    try:
        logging.info(f"Response Body: {json.dumps(response.json(), indent=2)}")
    except:
        logging.info(f"Response Body: {response.text}")

    return response


parent_id = str(uuid4())
thread_id = str(uuid4())
client = OpenAI(
    base_url="https://api.staging.langdb.ai/v1/",  # LangDB API base URL,
    api_key=api_key,  # Replace with your LangDB token
)


def make_request(messages, tools=None, parent_id=None, trace_id=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-parent-trace-id": parent_id,
        "x-trace-id": trace_id,
        "x-thread-id": thread_id,
    }

    payload = {"model": "gpt-4o-mini", "messages": messages, "tools": tools}

    return log_request("post", url, headers=headers, json=payload)


messages = [
    {
        "role": "system",
        "content": "You are a helful assistant. Forward the query to next assistant",
    },
    {
        "role": "user",
        "content": "What are the earnings of Apple in 2022?",
    },
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question to be asked",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

response = make_request(messages, tools=tools, trace_id=parent_id)

assistant_reply = response.json()["choices"][0]["message"]
print("Assistant:", assistant_reply)
tool_call_id = response.json()["choices"][0]["message"]["tool_calls"][0]["id"]
# Print the assistant's response
print("tool call id: ", tool_call_id)

print("#########################################")


def get_tool():
    """Returns the current time in H:MM AM/PM format."""
    messages = [
        {
            "role": "user",
            "content": "What are the earnings of Apple in 2022?",
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the model
        messages=messages,
        extra_headers={"x-parent-trace-id": parent_id},
    )
    return response.choices[0].message.content


x = get_tool()
print(x)


print("#########################################")
messages = [
    {
        "role": "system",
        "content": "You are a helful assistant. Forward the query to next assistant",
    },
    {
        "role": "user",
        "content": "What are the earnings of Apple in 2022?",
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call_id,
                "function": {
                    "name": "get_tool",
                    "arguments": '{"query": "What are the earnings of Apple in 2022?"}',
                },
                "type": "function",
            }
        ],
    },
    {"role": "tool", "tool_call_id": tool_call_id, "content": x},
]
response = make_request(messages, parent_id=parent_id)

print(response.json()["choices"][0]["message"]["content"])

print(parent_id)
