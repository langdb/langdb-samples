"""Web search and analysis agent - first step of 2-step web search process."""

import os
from google.adk.agents import LlmAgent
from uuid import uuid4
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types
from langdb_adk import LangDBLlm

from . import prompt

# Shared thread ID for the entire 2-step web search process
SHARED_THREAD_ID = str(uuid4())
SHARED_RUN_ID = str(uuid4())


def _render_reference(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    """Appends grounding references to the response."""
    del callback_context
    if (
        not llm_response.content or
        not llm_response.content.parts or
        not llm_response.grounding_metadata
    ):
        return llm_response
    references = []
    for chunk in llm_response.grounding_metadata.grounding_chunks or []:
        title, uri, text = '', '', ''
        if chunk.retrieved_context:
            title = chunk.retrieved_context.title
            uri = chunk.retrieved_context.uri
            text = chunk.retrieved_context.text
        elif chunk.web:
            title = chunk.web.title
            uri = chunk.web.uri
        parts = [s for s in (title, text) if s]
        if uri and parts:
            parts[0] = f'[{parts[0]}]({uri})'
        if parts:
            references.append('* ' + ': '.join(parts) + '\n')
    if references:
        reference_text = ''.join(['\n\nReference:\n\n'] + references)
        llm_response.content.parts.append(types.Part(text=reference_text))
    if all(part.text is not None for part in llm_response.content.parts):
        all_text = '\n'.join(part.text for part in llm_response.content.parts)
        llm_response.content.parts[0].text = all_text
        del llm_response.content.parts[1:]
    return llm_response

server_url = "https://api.us-east-1.langdb.ai/mcp/tavily_kq4hxxv"
critic_agent = LlmAgent(
    model=LangDBLlm(
        model="openai/gpt-4.1",
        api_key=os.getenv("LANGDB_API_KEY"),
        project_id = os.getenv("LANGDB_PROJECT_ID"),
        extra_headers={
            "x-thread-id": SHARED_THREAD_ID,
            "x-run-id": SHARED_RUN_ID
        },
        mcp_servers = [{
            "server_url": server_url,
            "type": "sse",
            "name": "Tavily"
        }]
    ),
    name="critic_agent",
    instruction=prompt.CRITIC_PROMPT,
    after_model_callback=_render_reference,
)
