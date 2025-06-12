"""
run.py

This is a multi-agent workflow that uses a query router to determine which agent to use based on the user's query.
It ends with a reply agent that formats the response with emojis.

There are multiple input and output guardrails to check the input and output of the agents.
"""

# Standard Library Imports
import asyncio
import os
import sys
import time
from typing import Literal
from dotenv import load_dotenv

load_dotenv()



# Third Party Imports
from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    Runner,
    WebSearchTool,
    set_default_openai_key,
    set_default_openai_client,
    set_tracing_disabled,
    OpenAIChatCompletionsModel
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    
)
from pydantic import BaseModel, Field
from uuid import uuid4


# ------------------------------------------------------------------------------------------------
# Initialize OpenAI client
# ------------------------------------------------------------------------------------------------

# Set default OpenAI key with use_for_tracing=False
set_tracing_disabled(True)
set_default_openai_key(key=os.environ.get("LANGDB_API_KEY"), use_for_tracing=False)
    

# Initialize with explicit parameters from environment variables
client = AsyncOpenAI(
    base_url=f"{os.environ.get('LANGDB_BASE_URL')}/{os.environ.get('LANGDB_PROJECT_ID')}/v1",
    api_key=os.environ.get("LANGDB_API_KEY"),
    default_headers = {
        "x-thread-id": str(uuid4()),
        "x-run-id": str(uuid4()),
    }
)


# Set the client for tracing
set_default_openai_client(client, use_for_tracing=False)


def get_model(model_name: str, client: AsyncOpenAI):
    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client
    )
    return model

# ------------------------------------------------------------------------------------------------
# Specialized Agents
# ------------------------------------------------------------------------------------------------
    
booking_agent = Agent(  
    name="Booking Specialist",
    model=get_model("gpt-4o", client),
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a booking specialist. You help customers with their booking and reservation questions.",
)

travel_recommendation_agent = Agent(
    name="Travel Recommendation Specialist",
    model=get_model("langdb/recc_8ac7wclb", client),  # langDB virtual model here, check readme for more information
    model_settings=ModelSettings(
        tool_choice='auto',
    ),
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a travel recommendation specialist. You help customers find ideal destinations and travel plans. Use the duckduckgo to find the best options.",
)

# ------------------------------------------------------------------------------------------------
# Orchestrator Agent (Tool-based approach)
# ------------------------------------------------------------------------------------------------

reply_agent = Agent(
    name="Reply Agent",
    model=get_model("langdb/reply_idzqgtrm", client),  # langDB virtual model here, check readme for more information
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You reply to the user's query and make it more informal by adding emojis.",
)

query_router_agent = Agent(
    name="Query Router",
    model=get_model("langdb/router_c77w2sji", client),  # langDB virtual model here, check readme for more information
    model_settings=ModelSettings(
        tool_choice='auto',
        parallel_tool_calls = False
    ),
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You determine which agent to use based on the user's query. "
        "If the query relates to booking flights, use the booking specialist. "
        "If the query relates to travel recommendations, use the travel recommendation specialist. "
        "Once you get the specialist response, always hand it off to the reply agent to format it with emojis."
    ),
    tools=[
        booking_agent.as_tool(
            tool_name="booking_tool",
            tool_description="Use when the user has questions about flight bookings, reservations, or ticketing",
        ),
        travel_recommendation_agent.as_tool(
            tool_name="travel_tool",
            tool_description="Use when the user wants travel destination recommendations or itinerary planning",
        )
    ],
    handoffs=[reply_agent],
)

async def main():
    # Get question from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python app.py 'Your question here'")
        return
    
    question = sys.argv[1]
    
    start_time = time.time()
    
    print("🔍 Processing your query: ", question)
    print("=" * 80)
    
    try:
        print("\nCreating RunConfig with custom provider...")
        run_config = RunConfig(
            tracing_disabled=True
        )
        print("RunConfig created, running agent...")
        
        result = Runner.run_streamed(
            starting_agent=query_router_agent, 
            input=question,
            run_config=run_config
        )
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                event_data = event.data
                if isinstance(event_data, ResponseCreatedEvent):
                    agent_name = result.last_agent.name
                    print(f"🏃 Starting `{agent_name}`")
                    print("-" * 50)
                elif isinstance(event_data, ResponseInProgressEvent):
                    print("⏳ Agent response in progress...")
                elif isinstance(event_data, ResponseOutputItemAddedEvent):
                    event_data_item = event_data.item
                    if isinstance(event_data_item, ResponseFunctionToolCall):
                        print(f"🔧 Tool called: {event_data_item.name}")
                        print("\t Arguments: ", end="")
                    elif isinstance(event_data_item, ResponseOutputMessage):
                        print("📝 Drafting response...")
                elif isinstance(event_data, ResponseFunctionCallArgumentsDeltaEvent):
                    event_data_delta = event_data.delta
                    print(event_data_delta, end="", flush=True)
                elif isinstance(event_data, ResponseFunctionCallArgumentsDoneEvent):
                    print("\n✅ Tool call completed!")
                elif isinstance(event_data, ResponseTextDeltaEvent):
                    print(event_data.delta, end="", flush=True)
            
            # Handle run_item_stream_event
            elif event.type == "run_item_stream_event":
                if event.name == "tool_output":
                    print("🛠️ Tool output:")
                    print("-" * 40)
            # Handle agent_updated_stream_event
            elif event.type == "agent_updated_stream_event":
                print("🔄 Agent Updated Event:")
                print(f"  New Agent: {event.new_agent.name}")
                print(f"  Instructions: {event.new_agent.instructions[:100]}...")
                print(f"  Tools: {[tool.name for tool in event.new_agent.tools]}")
                print("-" * 40)
            # Handle any other event types
            else:
                print(f"Unhandled event type: {event.type}")
                # Print the event details for debugging
                print(f"Event details: {vars(event)}")

    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        end_time = time.time()
        print("\n" + "=" * 80)
        print(f"✨ Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
