import uuid
from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from os import getenv
from dotenv import load_dotenv

load_dotenv()
# Initialize tools
tavily_tool = TavilySearchResults(tavily_api_key=getenv("TAVILY_API_KEY"), max_results=5)
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
) -> str:
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."

def make_system_prompt(suffix: str) -> str:
    """Create a system prompt with a custom suffix."""
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

def get_next_node(last_message: BaseMessage, goto: str) -> str:
    """Determine the next node based on the last message."""
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

def create_llm(api_base: str, project_id: str, model_name: str = "gpt-4", thread_id: str = str(uuid.uuid4())):
    """Create a ChatOpenAI instance with specified configuration."""
    return ChatOpenAI(
        model_name=model_name,
        openai_api_base=api_base,
        default_headers={"x-project-id": project_id, "x-thread-id": thread_id},
        api_key=getenv("LANGDB_API_KEY"),
    )



def create_workflow(llm_research, llm_chart):
    """Create and return a compiled workflow graph with initialized agents."""
    # Initialize agents
    research_agent = create_react_agent(
        llm_research,
        tools=[tavily_tool],
        state_modifier=make_system_prompt(
            "You can only do research. You are working with a chart generator colleague."
        ),
    )

    chart_agent = create_react_agent(
        llm_chart,
        [python_repl_tool],
        state_modifier=make_system_prompt(
            "You can only generate charts. You are working with a researcher colleague."
        ),
    )
    def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
        """Research node implementation."""
        result = research_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "chart_generator")
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="researcher"
        )
        return Command(
            update={"messages": result["messages"]},
            goto=goto,
        )

    def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
        """Chart generator node implementation."""
        result = chart_agent.invoke(state)
        goto = get_next_node(result["messages"][-1], "researcher")
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="chart_generator"
        )
        return Command(
            update={"messages": result["messages"]},
            goto=goto,
        )

    # Create workflow
    workflow = StateGraph(MessagesState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_edge(START, "researcher")
    return workflow.compile()

# Export necessary functions and tools
__all__ = [
    'create_llm',
    'create_workflow',
    'tavily_tool',
    'python_repl_tool',
]