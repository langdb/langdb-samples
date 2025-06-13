"""Web search and analysis agent - first step of 2-step web search process."""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from uuid import uuid4
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

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

critic_agent = LlmAgent(
    model=LiteLlm(
        "openai/openai/gpt-4.1",
        api_key=os.getenv("LANGDB_API_KEY"),
        api_base=f"{os.getenv('LANGDB_BASE_URL')}/{os.getenv('LANGDB_PROJECT_ID')}/v1",
        extra_headers={
            "x-thread-id": SHARED_THREAD_ID,
            "x-run-id": SHARED_RUN_ID
        }
    ),
    name="critic_agent",
    instruction=prompt.CRITIC_PROMPT,
    tools=[MCPToolset(
        connection_params=SseServerParams(
            url="https://api.staging.langdb.ai/mcp/a4588f1a-0366-4175-8757-f32820bbf2af",
            timeout=30,
        )
    )],
    after_model_callback=_render_reference,
)
