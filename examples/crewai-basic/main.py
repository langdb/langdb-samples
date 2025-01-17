from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.tools import BaseTool
from pydantic import Field
from os import getenv
from dotenv import load_dotenv
from uuid import uuid4
from typing import Annotated, Optional, Any, Type

load_dotenv()

PROJECT_ID = ""  # Replace with your LangDB project ID

default_headers = {
    "x-thread-id": str(uuid4()),
}

if PROJECT_ID:
    default_headers["x-project-id"] = PROJECT_ID

print(default_headers)


def main():
    llm_writer = LLM(
        model="gpt-4o-mini",
        base_url="https://api.us-east-1.langdb.ai",
        api_key=getenv("LANGDB_API_KEY"),
        extra_headers=default_headers,
    )
    writer_agent = Agent(
        role="Financial Writer",
        goal="Write insightful and engaging financial articles on various topics.",
        backstory="""Professional financial writer with expertise in economic trends, market analysis, and investment strategies.

    These keywords must never be transformed:
    - Action:
    - Thought:
    - Action Input:""",
        tools=[],  # No additional tools needed for writing
        llm=llm_writer,
        verbose=True,
    )
    # Define Writing Task with Dynamic Topic
    topic = "Impact of Rising Interest Rates on Global Markets"  # Example topic
    writing_task = Task(
        description=f"""Write a detailed financial article on the following topic:
    '{topic}'

    The article should:
    1. Have a clear and engaging introduction.
    2. Provide in-depth analysis and insights.
    3. Conclude with actionable takeaways or predictions.

    These keywords must never be transformed:
    - Action:
    - Thought:
    - Action Input:""",
        expected_output="""A comprehensive financial article with:
    1. Clear structure (introduction, body, conclusion).
    2. Relevant data and analysis.
    3. Engaging and professional tone.""",
        agent=writer_agent,
    )
    # Create Crew for Writing Task
    writer_crew = Crew(
        agents=[writer_agent],
        tasks=[writing_task],
        process=Process.sequential,
        verbose=True,
    )
    # Execute the writing task
    result = writer_crew.kickoff()
    print("Writer Agent Task Results:")
    print(result)


if __name__ == "__main__":
    main()
