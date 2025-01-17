from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import Field
from os import getenv, environ
from dotenv import load_dotenv
from uuid import uuid4
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


def create_llm(
    model_name: str,
    api_base: str = "https://api.us-east-1.langdb.ai",
    project_id: str = None,
    thread_id: str = None,
) -> LLM:
    """Create a LLM instance with specified configuration."""

    default_headers = {
        "content-type": "application/json",
    }

    if thread_id:
        default_headers["x-thread-id"] = thread_id

    if project_id:
        default_headers["x-project-id"] = project_id

    # Configure environment variables for OpenAI compatibility
    environ["OPENAI_API_BASE"] = api_base
    environ["OPENAI_API_KEY"] = getenv("LANGDB_API_KEY")

    return LLM(
        model=model_name,
        api_base=api_base,
        api_key=getenv("LANGDB_API_KEY"),
        extra_headers=default_headers,
    )


class TavilySearchTool(BaseTool):
    """Tool for performing web searches using Tavily API."""

    name: str = "Tavily Search"
    description: str = "Search for current market data and trends."
    search_tool: TavilySearchResults = Field(
        default_factory=lambda: TavilySearchResults(max_results=5)
    )

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search_tool.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"


class PythonREPLTool(BaseTool):
    """Tool for executing Python code, particularly useful for data analysis and visualization."""

    name: str = "Python REPL"
    description: str = """Execute Python code for data analysis and visualization. 
    Import required libraries before use. After creating a chart, write 'ANSWER: Created chart successfully'."""
    repl: PythonREPL = Field(default_factory=PythonREPL)

    def _run(self, command: str) -> str:
        """Execute python commands and return the output"""
        try:
            result = self.repl.run(command)
            if "plt.show()" in command or "plt.savefig(" in command:
                return "ANSWER: Created chart successfully"
            return result
        except Exception as e:
            return f"Error executing Python code: {str(e)}"


def create_researcher_agent(llm: LLM) -> Agent:
    """Create and return a Research Analyst agent."""
    return Agent(
        role="Research Analyst",
        goal="Find UK GDP data for the past 5 years",
        backstory="""Expert in finding economic statistics.

        These keywords must never be transformed:
        - Action:
        - Thought:
        - Action Input:""",
        tools=[TavilySearchTool()],
        llm=llm,
        verbose=True,
    )


def create_chart_generator_agent(llm: LLM) -> Agent:
    """Create and return a Data Visualization Expert agent."""
    return Agent(
        role="Data Visualization Expert",
        goal="Create GDP line chart",
        backstory="""Expert in creating charts with Python and matplotlib.

        These keywords must never be transformed:
        - Action:
        - Thought:
        - Action Input:""",
        tools=[PythonREPLTool()],
        llm=llm,
        verbose=True,
    )


def create_research_task(agent: Agent) -> Task:
    """Create and return the research task."""
    return Task(
        description="""Find UK GDP data (2019-2023) from reliable sources.
        Include annual values, years, and currency units.

        These keywords must never be transformed:
        - Action:
        - Thought:
        - Action Input:""",
        expected_output="""GDP data list with:
        1. Values (2019-2023)
        2. Currency units
        3. Source""",
        agent=agent,
    )


def create_chart_task(agent: Agent) -> Task:
    """Create and return the chart generation task."""
    return Task(
        description="""Create a line chart showing UK GDP trends (2019-2023).
        Use matplotlib, include proper labels and title.
        Save the chart as 'uk_gdp_chart.png'.

        These keywords must never be transformed:
        - Action:
        - Thought:
        - Action Input:""",
        expected_output="""Line chart with:
        1. GDP trend
        2. Clear labels
        3. Professional design
        4. Success message""",
        agent=agent,
    )


def create_crew(
    researcher_model: str = "gpt-4o",
    chart_model: str = "openai/claude-3-5-sonnet-20240620",
    project_id: str = None,
    process: Process = Process.sequential,
    verbose: bool = True,
) -> Crew:
    """
    Create and return a research crew with configured agents and tasks.

    Args:
        researcher_model: Model to use for the researcher agent
        chart_model: Model to use for the chart generator agent
        project_id: LangDB project ID
        process: Crew process type (sequential or parallel)
        verbose: Whether to enable verbose output

    Returns:
        Configured Crew instance
    """
    # Create LLMs
    thread_id = str(uuid4())
    researcher_llm = create_llm(
        researcher_model, project_id=project_id, thread_id=thread_id
    )
    chart_llm = create_llm(chart_model, project_id=project_id, thread_id=thread_id)

    # Create agents
    researcher = create_researcher_agent(researcher_llm)
    chart_generator = create_chart_generator_agent(chart_llm)

    # Create tasks
    research_task = create_research_task(agent=researcher)
    chart_task = create_chart_task(agent=chart_generator)

    # Create and return crew
    return Crew(
        agents=[researcher, chart_generator],
        tasks=[research_task, chart_task],
        process=process,
        verbose=verbose,
    )


# Export necessary functions and tools
__all__ = [
    "create_llm",
    "TavilySearchTool",
    "PythonREPLTool",
    "create_researcher_agent",
    "create_chart_generator_agent",
    "create_research_task",
    "create_chart_task",
    # "create_crew",
    # "create_youtube_tool",
    # "create_domain_expert_agent",
    # "create_content_writer_agent",
    # "create_youtube_research_task",
    # "create_article_writing_task",
    # "create_youtube_crew",
]
