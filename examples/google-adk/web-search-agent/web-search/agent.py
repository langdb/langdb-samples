"""2-step Web Search Agent for comprehensive web-based query responses."""

from google.adk.agents import SequentialAgent

from .sub_agents.critic import critic_agent
from .sub_agents.reviser import reviser_agent


llm_auditor = SequentialAgent(
    name='web_search_agent',
    description=(
        'A 2-step web search agent that first searches and analyzes web content,'
        ' then refines and synthesizes the information to provide comprehensive'
        ' answers to user queries.'
    ),
    sub_agents=[critic_agent, reviser_agent],
)

root_agent = llm_auditor
