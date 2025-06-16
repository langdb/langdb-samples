from openai import OpenAI
import time
import json

api_key = "langdb_NjVXcFI0SjZidmROcCtGa0ZKSVVVMVVYMnMwY2JjUGVmVy93elJjNEpOT3FNK0lQbWJ2aUZqWEZSb2M2TFZxQmozWUpOcy9kR0JIWS9TbmtOMktUVkdXem82bXJCOGMxeVJuVUdjazdFekh0QnpxZElTaFZOaVZJOHFhUTZrVGgzUUJXRlZTaUFqUS9Uc3pkY2Q2Y3BraXUwQVJ1WlcvNERWeWhMcTJ0KzVZZ0wzTHlyYXF4cW1UakFpc1ZTV05LRTVIREd3PT06QUFBQUFBQUFBQUFBQUFBQQ=="
PROJECT_ID = "10c717f2-0421-48b2-a20a-1269bb55829f"
api_base = f"https://api.staging.langdb.ai/{PROJECT_ID}/v1"

client = OpenAI(
    base_url=api_base,
    api_key=api_key,
)

questions = [
    "Explain how quantum entanglement works in simple terms.",
    "Write a Python function to implement merge sort algorithm.",
    "What are the key differences between REST and GraphQL APIs?",
    "Solve this math problem: If a train travels at 60 mph for 2.5 hours, then at 75 mph for 1.5 hours, what's the total distance covered?",
    "Write a haiku about artificial intelligence.",
    "Explain the concept of database indexing and when to use it.",
    "Debug this code snippet: for i in range(10): print(i] print('Done')",
    "What are the main principles of clean architecture in software development?",
    "Explain how LSTM networks work in deep learning.",
    "Design a system architecture for a real-time chat application.",
]


def get_model_response(question, model_name):
    try:
        messages = [{"role": "user", "content": question}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def compare_models():
    results = []
    models = ["deepseek-reasoner", "o1-preview"]

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 80)

        comparison = {"question": question, "responses": {}}

        for model in models:
            print(f"\nTesting {model}...")
            start_time = time.time()
            response = get_model_response(question, model)
            end_time = time.time()

            comparison["responses"][model] = {
                "response": response,
                "time_taken": round(end_time - start_time, 2),
            }

            print(f"{model} Response:")
            print(response)
            print(f"Time taken: {round(end_time - start_time, 2)} seconds")

        results.append(comparison)

    # Save results to a JSON file
    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    print("Starting model comparison...")
    compare_models()
    print(
        "\nComparison complete! Results have been saved to model_comparison_results.json"
    )
