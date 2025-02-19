# Banking RAG Chatbot with LangDB.ai & LangChain

Welcome to the Banking FAQ Assistant repository! This project demonstrates how to build a scalable, RAG-powered conversational AI using **LangDB.ai** and **LangChain**. Perfect for creating intelligent banking chatbots that can answer questions about loans, interest rates, and general banking services.

## Features

- **RAG-Powered Responses:** Leverages Retrieval-Augmented Generation for accurate banking information
- **Seamless LangDB.ai Integration:** Uses LangDB.ai's optimized API endpoints for LLM interactions
- **Streamlit UI:** Clean and responsive chat interface
- **Conversation Memory:** Maintains context throughout the chat session
- **Advanced AI Observability:** Leverage AI analytics and operational visibility through LangDB.ai
- **Free Starter Credit:** Begin with $10 free credit from LangDB.ai!

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/langdb/langdb-samples.git
   cd langdb-samples
   ```

2. **Set Up Your Environment:**
   ```bash
   conda create -n banking-bot python=3.10
   conda activate banking-bot
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   ```bash
   # For Windows:
   set LANGDB_API_KEY=your-langdb-api-key

   # For macOS/Linux:
   export LANGDB_API_KEY=your-langdb-api-key
   ```

4. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

## Project Structure

```
examples/
└── langchain/
    └── langchain-rag-bot/
        ├── main.py            # Main application file
        ├── requirements.txt   # Project dependencies
        └── README.md          # Documentation
```

## Configuration

Update the following constants in `main.py`:

```python
LANGDB_API_URL = "https://api.us-east-1.langdb.ai/your-project-id/v1"
```

## Usage

1. Start the application using Streamlit
2. Enter banking-related questions in the chat interface
3. Receive AI-powered responses based on the banking context
4. The conversation history is maintained throughout the session

## Getting Started

To begin using this chatbot, check out our [Blog](https://blog.langdb.ai) for detailed tutorials and best practices. Don't forget to claim your $10 free credit from LangDB.ai to experiment with these features!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

Build smarter banking interactions with AI! 🏦 🤖