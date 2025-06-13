
"""Synthesis and refinement agent - second step of 2-step web search process."""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from . import prompt
from ..critic.agent import SHARED_THREAD_ID, SHARED_RUN_ID

_END_OF_EDIT_MARK = "---END-OF-EDIT---"


def _remove_end_of_edit_mark(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    del callback_context  # unused
    if not llm_response.content or not llm_response.content.parts:
        return llm_response
    for idx, part in enumerate(llm_response.content.parts):
        if _END_OF_EDIT_MARK in part.text:
            del llm_response.content.parts[idx + 1 :]
            part.text = part.text.split(_END_OF_EDIT_MARK, 1)[0]
    return llm_response

reviser_agent = LlmAgent(
    model=LiteLlm(
        "openai/openai/gpt-4.1",
        api_key=os.getenv("LANGDB_API_KEY"),
        api_base=f"{os.getenv('LANGDB_BASE_URL')}/{os.getenv('LANGDB_PROJECT_ID')}/v1",
        extra_headers={
            "x-thread-id": SHARED_THREAD_ID,
            "x-run-id": SHARED_RUN_ID
        }
    ),
    name="reviser_agent",
    instruction=prompt.REVISER_PROMPT,
    after_model_callback=_remove_end_of_edit_mark,
)
