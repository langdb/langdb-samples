# LlamaIndex Basic Example
This example demonstrates how to use LlamaIndex to create a document index and query it using OpenAI's GPT model via LangDB.

## Prerequisites
- Python 3.8+
- Access to LangDB.ai API
- LangDB.ai API key

## Installation
```bash
pip install llama-index openai
```
## Setup
    - Create a data directory in the same folder as main.py and add your documents there.

## Code Explanation
The main.py script:

1. Initializes LlamaIndex with LangDB.ai configuration
2. Loads documents from the data directory consisting of langdb.ai features
3. Creates a vector store index from the documents
4. Persists the index to disk in a storage directory
5. Creates a query engine to interact with the index
6. Performs a sample query about LangDB.ai features

## Usage
1. Place your documents in the data directory
2. Run the script:
```bash
python main.py
```

## Key Components
- `SimpleDirectoryReader`: Loads documents from a directory
- `VectorStoreIndex`: Creates and manages document embeddings
- `OpenAI`: LLM configuration for queries
- `query_engine`: Interface for querying the indexed documents

## Output
The script will print the response to the query "what are the features of langdb.ai?" based on the content of your indexed documents.
