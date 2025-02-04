from crewai.flow.flow import Flow, listen, start
from crewai import Agent, Task, Crew, LLM
from crewai_tools import WebsiteSearchTool
from pydantic import BaseModel
import os
from openai import OpenAI
from uuid import uuid4
from utils import completion, base_url
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


load_dotenv()


def crewai_llm(model="gpt-4o-mini", headers=None, extra_body=None):
    return LLM(
        model=model,
        base_url=base_url,
        extra_headers=headers,
        api_key=os.getenv("LANGDB_API_KEY"),
        extra_body=extra_body,
    )


def openai_client(
    api_key=os.getenv("LANGDB_API_KEY"),
    headers=None,
    base_url=base_url,
):
    return OpenAI(base_url=base_url, api_key=api_key, default_headers=headers)


def create_qa_flow(
    q,
    first_model="gpt-4o-mini",
    second_model="gpt-4o-mini",
    tags: str = None,
    routing_enabled=False,
    thread_id=str(uuid4()),
    project_id=None,
):
    default_headers = {"x-thread-id": thread_id, "x-tags": tags}
    if project_id:
        default_headers["x-project-id"] = project_id
    if routing_enabled:
        model = "openai/router/dynamic"
        extra_body = {
            "extra": {
                "name": "Test",
                "strategy": {"type": "cost", "willingness_to_pay": 0.5},
                "models": [first_model, second_model],
            }
        }
    else:
        model = first_model
    client1 = openai_client(headers=default_headers)
    llm = crewai_llm(
        model=model,
        headers=default_headers,
        extra_body=extra_body if routing_enabled else None,
    )

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    research_agent = Agent(
        role="You are a helpful assistant that can answer questions about the web.",
        goal="Answer the user's question.",
        backstory="You have access to a vast knowledge base of information from the web.",
        tools=[
            WebsiteSearchTool(website=urls[0]),
            WebsiteSearchTool(website=urls[1]),
            WebsiteSearchTool(website=urls[2]),
        ],
        llm=llm,
        use_system_prompt=False,
    )

    task = Task(
        description="Answer the following question: {question}",
        expected_output="A detailed and accurate answer to the user's question.",
        agent=research_agent,
    )

    crew = Crew(
        agents=[research_agent],
        tasks=[task],
    )

    class QAState(BaseModel):
        question: str = q
        """
    State for the documentation flow
    """
        improved_question: str = ""
        answer: str = ""

    class QAFlow(Flow[QAState]):
        client = client1
        model = second_model

        @start()
        def rewrite_question(self):
            print(f"# Rewriting question: {self.state.question}")
            routing_extra_body = None
            if routing_enabled:
                self.model = "router/dynamic"
                routing_extra_body = {
                    "extra": {
                        "name": "Test",
                        "strategy": {"type": "cost", "willingness_to_pay": 0.5},
                        "models": [first_model, second_model],
                    }
                }

            messages = [
                {
                    "role": "user",
                    "content": f"""Look at the input and try to reason about the underlying semantic intent / meaning.
              Here is the initial question:
              -------
              {self.state.question}
              -------
              Formulate an improved question:""",
                }
            ]

            improved_question = completion(
                client=self.client,
                model=self.model if not routing_enabled else "router/dynamic",
                messages=messages,
                extra_body=routing_extra_body,
            )

            print(improved_question)
            self.state.improved_question = improved_question

        @listen(rewrite_question)
        def answer_question(self):
            print(f"# Answering question: {self.state.improved_question}")
            result = crew.kickoff(inputs={"question": self.state.improved_question})
            self.state.answer = result.raw
            return result

    return QAFlow()


def rate_answers(csv_path: str, output_csv_path: str = None):
    """
    Reads a CSV containing at least:
      - question
      - answer_single
      - answer_routed
    Uses an LLM to score each answer on a scale of 1-10,
    and stores those scores in single_score and routed_score columns.

    Args:
        csv_path: Path to input CSV file
        output_csv_path: If provided, saves results to a new CSV file
                         rather than overwriting the original.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create an LLM client (example with OpenAI)
    client = openai_client()  # Or however you initialize your LLM client

    # Define the scoring prompt template
    scoring_prompt = """
    You are an expert evaluator. Rate the following answer to a question about LLMs and autonomous agents.
    Rate the answer on a scale of 1-10 based on:
    - Accuracy and correctness (4 points)
    - Completeness and depth (3 points)
    - Clarity and coherence (3 points)

    Question: {question}
    Answer: {answer}

    Provide only the numerical score without any explanation.
    """

    # Ensure columns for scores exist
    if "single_score" not in df.columns:
        df["single_score"] = 0.0
    if "routed_score" not in df.columns:
        df["routed_score"] = 0.0

    # Iterate over each row to rate answers
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rating answers"):
        question_text = str(row["question"])

        # ------------------- Single Score -------------------
        single_answer_text = str(row["single_answer"])

        # Build messages for the scoring LLM
        messages_single = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Provide only a numerical score.",
            },
            {
                "role": "user",
                "content": scoring_prompt.format(
                    question=question_text, answer=single_answer_text
                ),
            },
        ]

        try:
            single_score_resp = completion(client, "gpt-4o", messages_single)
            # Extract just the numeric score from the model's response
            single_score_str = single_score_resp
            df.at[idx, "single_score"] = float(single_score_str)
        except Exception as e:
            print(f"[Single] Error scoring row {idx}: {e}")
            # Optionally leave the default 0.0 or handle differently
            df.at[idx, "single_score"] = 0.0

        # ------------------- Routed Score -------------------
        routed_answer_text = str(row["routed_answer"])

        messages_routed = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Provide only a numerical score.",
            },
            {
                "role": "user",
                "content": scoring_prompt.format(
                    question=question_text, answer=routed_answer_text
                ),
            },
        ]

        try:
            routed_score_resp = completion(client, "gpt-4o", messages_routed)
            routed_score_str = routed_score_resp
            df.at[idx, "routed_score"] = float(routed_score_str)
        except Exception as e:
            print(f"[Routed] Error scoring row {idx}: {e}")
            df.at[idx, "routed_score"] = 0.0

    # Save back to CSV
    out_path = output_csv_path if output_csv_path else csv_path
    df.to_csv(out_path, index=False)
    return df


def plot_cost_comparison_log(csv_path):
    """
    Plot a line graph comparing the costs of single and routed answers from a CSV file in dark mode.

    Args:
        csv_path (str): Path to the CSV file containing the data.

    The CSV must contain the following columns:
        - 'cost_single': Costs for single answers.
        - 'cost_routed': Costs for routed answers.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Check if the required columns are present
        required_columns = ["single_cost", "routed_cost"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"The CSV must contain the following columns: {required_columns}"
            )

        # Plotting the cost comparison with a dark theme
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 6))
        plt.plot(
            df.index,
            df["single_cost"],
            label="Cost Direct",
            color="#1E3A8A",
            linewidth=2,
        )
        plt.plot(
            df.index,
            df["routed_cost"],
            label="Cost Routed",
            color="#B91C1C",
            linewidth=2,
        )
        plt.yscale("log")
        # Adding labels, title, and legend
        plt.xlabel("Question Index", fontsize=12, color="white")
        plt.ylabel("LLM Cost in USD", fontsize=12, color="white")
        plt.title(
            "Cost Comparison: Direct vs. Routed Workflows", fontsize=14, color="white"
        )
        plt.legend(fontsize=10, facecolor="gray", edgecolor="white")
        plt.grid(color="gray", linestyle="--", linewidth=0.5)

        # Adjusting layout
        plt.tight_layout()

        # Save the plot with a proper filename
        output_filename = (
            f"./cost_comparison_{os.path.basename(csv_path).replace('.csv', '')}.png"
        )
        plt.savefig(output_filename)

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except ValueError as ve:
        print(str(ve))
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_cost_comparison(csv_path):
    """
    Plot a line graph comparing the costs of single and routed answers from a CSV file in dark mode.

    Args:
        csv_path (str): Path to the CSV file containing the data.

    The CSV must contain the following columns:
        - 'cost_single': Costs for single answers.
        - 'cost_routed': Costs for routed answers.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Check if the required columns are present
        required_columns = ["single_cost", "routed_cost"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"The CSV must contain the following columns: {required_columns}"
            )

        # Plotting the cost comparison with a dark theme
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 6))
        plt.plot(
            df.index,
            df["single_cost"],
            label="Cost Direct",
            color="#1E3A8A",
            linewidth=2,
        )
        plt.plot(
            df.index,
            df["routed_cost"],
            label="Cost Routed",
            color="#B91C1C",
            linewidth=2,
        )

        # Adding labels, title, and legend
        plt.xlabel("Question Index", fontsize=12, color="white")
        plt.ylabel("LLM Cost in USD", fontsize=12, color="white")
        plt.title(
            "Cost Comparison: Direct vs. Routed Workflows", fontsize=14, color="white"
        )
        plt.legend(fontsize=10, facecolor="gray", edgecolor="white")
        plt.grid(color="gray", linestyle="--", linewidth=0.5)

        # Adjusting layout
        plt.tight_layout()

        # Save the plot with a proper filename
        output_filename = (
            f"./cost_comparison_{os.path.basename(csv_path).replace('.csv', '')}.png"
        )
        plt.savefig(output_filename)

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except ValueError as ve:
        print(str(ve))
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_cost_difference(csv_path):
    """
    Plot a line graph showing the cost difference between single and routed answers from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing the data.

    The CSV must contain the following columns:
        - 'single_cost': Costs for single answers.
        - 'routed_cost': Costs for routed answers.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Check if the required columns are present
        required_columns = ["single_cost", "routed_cost"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"The CSV must contain the following columns: {required_columns}"
            )

        # Calculate the cost difference
        df["cost_difference"] = df["single_cost"] - df["routed_cost"]

        # Plotting the cost difference
        plt.figure(figsize=(10, 6))
        plt.bar(
            df.index, df["cost_difference"], color="skyblue", label="Cost Difference"
        )

        # Adding labels, title, and legend
        plt.xlabel("Index")
        plt.ylabel("Cost Difference")
        plt.title("Cost Difference Between Single and Routed Answers")
        plt.axhline(
            0, color="gray", linestyle="--", linewidth=0.8
        )  # Reference line for zero difference
        plt.legend()
        plt.grid(axis="y", linestyle="--", linewidth=0.7)
        plt.tight_layout()

        # Display the plot
        plt.show()
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except ValueError as ve:
        print(str(ve))
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_cost_difference_log(csv_path):
    """
    Plot a line graph showing the cost difference between single and routed answers from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing the data.

    The CSV must contain the following columns:
        - 'single_cost': Costs for single answers.
        - 'routed_cost': Costs for routed answers.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Check if the required columns are present
        required_columns = ["single_cost", "routed_cost"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"The CSV must contain the following columns: {required_columns}"
            )

        # Calculate the cost difference
        df["cost_difference"] = df["single_cost"] - df["routed_cost"]

        # Plotting the cost difference
        plt.figure(figsize=(10, 6))
        plt.bar(
            df.index, df["cost_difference"], color="skyblue", label="Cost Difference"
        )

        # Adding labels, title, and legend
        plt.xlabel("Index")
        plt.ylabel("Cost Difference")
        plt.title("Cost Difference Between Single and Routed Answers")
        plt.axhline(
            0, color="gray", linestyle="--", linewidth=0.8
        )  # Reference line for zero difference
        plt.legend()
        plt.grid(axis="y", linestyle="--", linewidth=0.7)
        plt.tight_layout()

        # Display the plot
        plt.show()
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except ValueError as ve:
        print(str(ve))
    except Exception as e:
        print(f"An error occurred: {e}")


def compare_scores(
    csv_path: str, single_col: str = "single_score", routed_col: str = "routed_score"
) -> dict:
    """
    Compare scores between two columns (e.g., single vs. routed).

    Args:
        csv_path: Path to the CSV file containing your scoring columns
        single_col: Column name for the "single" approach scores
        routed_col: Column name for the "routed" approach scores

    Returns:
        A dictionary with comparison statistics, for example:
        {
          "mean_single": float,
          "mean_routed": float,
          "avg_difference": float,  # (routed - single)
          "times_routed_higher": int,
          "times_single_higher": int,
          "times_equal": int,
          "min_difference": float,
          "max_difference": float
        }
    """
    # Ensure the columns exist
    df = pd.read_csv(csv_path)
    if single_col not in df.columns or routed_col not in df.columns:
        raise ValueError(
            f"DataFrame must have columns '{single_col}' and '{routed_col}'"
        )

    # Drop rows with missing scores to avoid NaN issues
    temp_df = df.dropna(subset=[single_col, routed_col]).copy()

    # Basic means
    mean_single = temp_df[single_col].mean()
    mean_routed = temp_df[routed_col].mean()

    # Differences: (routed - single)
    temp_df["difference"] = temp_df[routed_col] - temp_df[single_col]
    avg_difference = temp_df["difference"].mean()
    min_difference = temp_df["difference"].min()
    max_difference = temp_df["difference"].max()

    # Count how many times routed is higher vs. single
    times_routed_higher = (temp_df["difference"] > 0).sum()
    times_single_higher = (temp_df["difference"] < 0).sum()
    times_equal = (temp_df["difference"] == 0).sum()

    # Package results
    results = {
        "mean_single": mean_single,
        "mean_routed": mean_routed,
        "avg_difference": avg_difference,
        "times_routed_higher": times_routed_higher,
        "times_single_higher": times_single_higher,
        "times_equal": times_equal,
        "min_difference": min_difference,
        "max_difference": max_difference,
    }
    return results


def calculate_total_costs(csv_path):
    """
    Calculate the total costs for both single and routed approaches from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing the data

    Returns:
        tuple: (total_single_cost, total_routed_cost, cost_difference, percentage_savings)
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Calculate totals
        total_single_cost = df["single_cost"].sum()
        total_routed_cost = df["routed_cost"].sum()

        # Calculate difference and percentage savings
        cost_difference = total_single_cost - total_routed_cost
        percentage_savings = (
            (cost_difference / total_single_cost) * 100 if total_single_cost > 0 else 0
        )

        print(f"\nCost Analysis for {os.path.basename(csv_path)}:")
        print(f"Total Direct Cost: ${total_single_cost:.4f}")
        print(f"Total Routed Cost: ${total_routed_cost:.4f}")
        print(f"Cost Savings: ${cost_difference:.4f}")
        print(f"Percentage Savings: {percentage_savings:.2f}%")

        return total_single_cost, total_routed_cost, cost_difference, percentage_savings

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None


if __name__ == "__main__":
    flow = create_qa_flow(
        "What does Lilian Weng say about the types of agent memory?",
        second_model="gpt-4o-mini",
        tags="flow=qa_flow",
    )
    result = flow.kickoff()
    print("=" * 10)
    print(result)


def plot_cost_comparison_cover(csv_path):
    """
    Plot a line graph comparing the costs of single and routed answers from a CSV file in dark mode.

    Args:
        csv_path (str): Path to the CSV file containing the data.

    The CSV must contain the following columns:
        - 'single_cost': Costs for single answers.
        - 'routed_cost': Costs for routed answers.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)

        # Check if the required columns are present
        required_columns = ["single_cost", "routed_cost"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"The CSV must contain the following columns: {required_columns}"
            )

        # Plotting the cost comparison with a dark theme
        plt.style.use("dark_background")
        plt.figure(figsize=(15, 8), dpi=300)  # Dimensions: 1600x800 pixels
        plt.plot(
            df.index,
            df["single_cost"],
            label="Cost Direct",
            color="#1E3A8A",
            linewidth=2,
        )
        plt.plot(
            df.index,
            df["routed_cost"],
            label="Cost Routed",
            color="#B91C1C",
            linewidth=2,
        )
        plt.yscale("log")

        # Adding labels, title, and legend
        plt.xlabel("Question Index", fontsize=12, color="white")
        plt.ylabel("LLM Cost in USD", fontsize=12, color="white")
        plt.title(
            "Cost Comparison: Direct vs. Routed Workflows", fontsize=14, color="white"
        )
        plt.legend(fontsize=10, facecolor="gray", edgecolor="white")
        plt.grid(color="gray", linestyle="--", linewidth=0.5)

        # Adjusting layout
        plt.tight_layout()

        # Save the plot with a proper filename
        output_filename = (
            f"./cover_plots_{os.path.basename(csv_path).replace('.csv', '')}.png"
        )
        plt.savefig(output_filename)
        print(f"Plot saved as {output_filename}")

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except ValueError as ve:
        print(str(ve))
    except Exception as e:
        print(f"An error occurred: {e}")
