from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.tavily_search.tool import TavilySearchResults
import uuid
from dotenv import load_dotenv
from os import getenv

load_dotenv()

PROJECT_ID = ""  ## LangDB Project ID

def get_function_tools():
    tavily_tool = TavilySearchResults(
        tavily_api_key=getenv("TAVILY_API_KEY"), max_results=5
    )

    tools = [tavily_tool]
    tools.extend(load_tools(["wikipedia"]))

    return tools


def create_llm(
    api_base: str,
    project_id: str,
    model_name: str = "gpt-4o-mini",
    thread_id: str = str(uuid.uuid4()),
):
    """Create a ChatOpenAI instance with specified configuration."""
    if not project_id:
        default_headers = {
            "x-thread-id": thread_id
        }
    else:
        default_headers = {
            "x-project-id": project_id,
            "x-thread-id": thread_id
        }
    return ChatOpenAI(
        model_name=model_name,
        openai_api_base=api_base,
        default_headers=default_headers,
        api_key=getenv("LANGDB_API_KEY"),
    )


def init_action(question, model):
    llm = create_llm(
        api_base="https://api.us-east-1.langdb.ai",
        project_id=PROJECT_ID,
        model_name=model,
    )
    prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = get_function_tools()
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": question})


init_action("Tell me about the owner of Tesla Company", "gpt-4o-mini")
