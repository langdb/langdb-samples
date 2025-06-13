# Google ADK Web Search Agent

A comprehensive 2-step web search agent system built with Google ADK that provides thorough research and synthesized answers to user queries through sequential agent processing.

## Overview

This application uses a two-step sequential agent approach:
- **Critic Agent**: Conducts comprehensive web searches and analyzes information
- **Reviser Agent**: Synthesizes research findings into well-structured, comprehensive answers

The agents work in sequence with a shared thread ID to maintain context throughout the research and synthesis process.

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# LangDB API Configuration
LANGDB_API_KEY=your_api_key_here
LANGDB_BASE_URL=your_base_url_here
LANGDB_PROJECT_ID=your_project_id_here
```

### Environment Variable Details

- `LANGDB_API_KEY`: Your LangDB API key (required)
- `LANGDB_BASE_URL`: The base URL for your LangDB instance
- `LANGDB_PROJECT_ID`: Your LangDB project identifier

## Model Configuration

Models are configured individually for each agent in the sub-agent files. Each agent uses LiteLLM to configure its language model through LangDB virtual models.

> We need to setup Virtual MCP Server and Virtual Model for the `Critic Agent` to use Tavily Search MCP for web search capabilities.

## MCP Server Integration Options

You have two options for integrating MCP servers with the web search agent:

### Option 1: Direct MCP Server Connection (Current Implementation)

The current implementation dynamically connects to an MCP server via MCPToolset using a utility function:

```python
# Dynamic MCP URL generation
from ...utils import get_dynamic_mcp_url

# Get dynamic MCP server URL
mcp_slug = "search_7l9zk5zp"  # Your MCP Server Slug
mcp_url = get_dynamic_mcp_url(mcp_slug)
if not mcp_url:
    raise RuntimeError("Failed to get dynamic MCP server URL. Check environment variables and API connectivity.")

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
            url=mcp_url,  # Dynamic URL
            timeout=30,
        )
    )],
    after_model_callback=_render_reference,
)
```

#### Dynamic MCP URL Generation

The system automatically creates MCP server sessions using the `utils.py` helper:

- **`get_dynamic_mcp_url(mcp_slug)`**: Creates a session by POSTing to `{host}/mcp-servers/{mcp_slug}/session`
- **Authentication**: Uses `LANGDB_API_KEY` as Bearer token
- **Session URL**: Returns `{host}/mcp/{session_id}` for the MCPToolset
- **Error Handling**: Throws `RuntimeError` if session creation fails

To use with your own MCP server, update the `mcp_slug` variable in `critic/agent.py`.

### Option 2: Virtual Model with Attached MCP Server

Alternatively, you can create a virtual model with an attached MCP server:

#### Step 1: Virtual MCP Server Setup

1. Log in and navigate to **MCP Servers › Virtual MCP Servers** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual MCP Server** and configure:
   - **Name**: e.g. `web-search-mcp`  
   - **Underlying MCP**: choose [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9)

#### Step 2: Virtual Model Setup

1. Log in and navigate to **Project › Models** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual Model** and configure name, base model, version, and MCP Server for search tools. Use the [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9) you created above.
3. Copy the generated model name (e.g. `openai/langdb/web-search-critic@v1`).  
4. Update your `critic/agent.py` to use the virtual model name and remove the MCPToolset:

```python
critic_agent = LlmAgent(
    model=LiteLlm(
        "openai/langdb/your-model-name",  # Your LangDB virtual model here
        api_key=os.getenv("LANGDB_API_KEY"),
        api_base=f"{os.getenv('LANGDB_BASE_URL')}/{os.getenv('LANGDB_PROJECT_ID')}/v1",
        extra_headers={
            "x-thread-id": SHARED_THREAD_ID,
            "x-run-id": SHARED_RUN_ID
        }
    ),
    name="critic_agent",
    instruction=prompt.CRITIC_PROMPT,
    # No need for tools parameter - MCP server is attached to the virtual model
    after_model_callback=_render_reference,
)
```

## Architecture

The web search agent uses a **SequentialAgent** architecture with two specialized sub-agents:

1. **Critic Agent** (with MCP tools for web search)
   - Analyzes user queries to understand information needs
   - Conducts comprehensive web searches using multiple search strategies
   - Evaluates source reliability and information quality
   - Organizes findings and identifies information gaps

2. **Reviser Agent** (synthesis and formatting)
   - Receives research findings from the Critic Agent
   - Synthesizes information into coherent, comprehensive answers
   - Structures responses with appropriate formatting and organization
   - Ensures final answers directly address user queries

Both agents share a common thread ID to maintain context throughout the process.

## Key Features

### 1. Sequential Processing
- Two-step approach ensures thorough research followed by quality synthesis
- Shared thread ID maintains context between agents

### 2. Comprehensive Web Search
- Multiple search strategies and varied search terms
- Focus on authoritative and recent sources
- Analysis of conflicting information and source reliability

### 3. Virtual MCP Server Integration
- Web search capabilities through Tavily Search MCP
- Seamless integration with LangDB virtual models
- Configurable search tools and parameters

### 4. Professional Synthesis
- Well-structured, comprehensive final answers
- Acknowledgment of limitations and conflicting information
- Clear, accessible writing style with proper organization

### 5. Model Flexibility
- LiteLLM integration supports multiple model providers
- Configurable model selection per agent
- LangDB virtual model support for enhanced capabilities

## Usage

### Example Usage

```bash
adk run web-search
```
Or 
```bash
adk web
```

## Project Structure

```
web-search-agent/
├── web-search/
│   ├── __init__.py
│   ├── agent.py              # Main SequentialAgent configuration
│   └── sub_agents/
│       ├── __init__.py
│       ├── critic/           # Web search and analysis agent
│       │   ├── __init__.py
│       │   ├── agent.py      # Critic agent with MCP tools
│       │   └── prompt.py     # Research and analysis prompts
│       └── reviser/          # Synthesis and formatting agent
│           ├── __init__.py
│           ├── agent.py      # Reviser agent configuration
│           └── prompt.py     # Synthesis and writing prompts
├── pyproject.toml            # Project dependencies
└── README.md           # This file
```

## Dependencies

### Prerequisites

- Python 3.11+
- Google ADK (Agents Development Kit)
- LiteLLM for model integration

### Installation

```bash
pip install google-adk litellm
```

### Main Dependencies

- `google-adk`: Google's Agents Development Kit for multi-agent systems
- `litellm`: Unified interface for various LLM providers

## Model Configuration Details

### Current Configuration

Both agents are configured to use GPT-4.1 through LangDB:

```python
# Critic Agent (with web search tools)
model=LiteLlm(
    "openai/openai/gpt-4.1",
    api_key=os.getenv("LANGDB_API_KEY"),
    api_base=f"{os.getenv('LANGDB_BASE_URL')}/{os.getenv('LANGDB_PROJECT_ID')}/v1"
)

# Reviser Agent (synthesis only)
model=LiteLlm(
    "openai/openai/gpt-4.1",
    api_key=os.getenv("LANGDB_API_KEY"),
    api_base=f"{os.getenv('LANGDB_BASE_URL')}/{os.getenv('LANGDB_PROJECT_ID')}/v1"
)
```

### Supported Model Formats

- OpenAI models: `"openai/gpt-4.1"`, `"openai/gpt-3.5-turbo"`
- Other Provider Models: Follow LiteLLM format `"openai/anthropic/claude-sonnet-4"`
- LangDB Virtual Models: `"openai/langdb/your-model-name"`

## =� References

* [Google ADK Documentation](https://developers.google.com/agent-development-kit)
* [LangDB Virtual MCP Servers](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [LangDB Virtual Models](https://docs.langdb.ai/concepts/virtual-models)
* [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9)
* [LiteLLM Documentation](https://docs.litellm.ai/)

---

Enjoy building comprehensive web search workflows with Google ADK + LangDB! Configure your virtual MCP servers and models through the LangDB dashboard for optimal search capabilities.