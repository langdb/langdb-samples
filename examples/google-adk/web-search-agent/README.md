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
- `LANGDB_PROJECT_ID`: Your LangDB project identifier

## Model Configuration

Models are configured individually for each agent in the sub-agent files. Each agent uses LangDBLlm from the langdb-adk package (an extension of Google ADK) to configure its language model with direct LangDB and MCP server integration.

> The `Critic Agent` is configured with Tavily Search MCP server for web search capabilities through LangDB's direct MCP server configuration.

## MCP Server Integration

The current implementation uses Google ADK with the langdb-adk extension, which provides LangDBLlm for direct LangDB and MCP server integration. This approach simplifies the integration by configuring MCP servers directly in the model initialization:

```python
from langdb_adk import LangDBLlm

server_url = "https://api.us-east-1.langdb.ai/mcp/tavily_wtok4wiw"
critic_agent = LlmAgent(
    model=LangDBLlm(
        model="openai/gpt-4.1",
        api_key=os.getenv("LANGDB_API_KEY"),
        project_id=os.getenv("LANGDB_PROJECT_ID"),
        extra_headers={
            "x-thread-id": SHARED_THREAD_ID,
            "x-run-id": SHARED_RUN_ID
        },
        mcp_servers=[{
            "server_url": server_url,
            "type": "sse",
            "name": "Tavily"
        }]
    ),
    name="critic_agent",
    instruction=prompt.CRITIC_PROMPT,
    after_model_callback=_render_reference,
)
```

### MCP Server Configuration

- **Direct Integration**: MCP servers are configured directly in the LangDBLlm model initialization
- **Server URL**: Use your LangDB MCP server URL (e.g., `https://api.us-east-1.langdb.ai/mcp/your_server_id`)
- **Authentication**: Uses `LANGDB_API_KEY` for authentication
- **Type**: Specify connection type ("sse" for Server-Sent Events)

To use with your own MCP server, update the `server_url` variable in `critic/agent.py`.

## Alternative: Virtual Model with Attached MCP Server

You can also create a virtual model with an attached MCP server:

#### Step 1: Virtual MCP Server Setup

1. Log in and navigate to **MCP Servers › Virtual MCP Servers** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual MCP Server** and configure:
   - **Name**: e.g. `web-search-mcp`  
   - **Underlying MCP**: choose [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9)

#### Step 2: Virtual Model Setup

1. Log in and navigate to **Project › Models** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual Model** and configure name, base model, version, and MCP Server for search tools. Use the [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9) you created above.
3. Copy the generated model name (e.g. `openai/langdb/web-search-critic@v1`).  
4. Update your `critic/agent.py` to use the virtual model name:

```python
critic_agent = LlmAgent(
    model=LangDBLlm(
        model="openai/langdb/your-model-name",  # Your LangDB virtual model here
        api_key=os.getenv("LANGDB_API_KEY"),
        project_id=os.getenv("LANGDB_PROJECT_ID"),
        extra_headers={
            "x-thread-id": SHARED_THREAD_ID,
            "x-run-id": SHARED_RUN_ID
        }
        # No need for mcp_servers parameter - MCP server is attached to the virtual model
    ),
    name="critic_agent",
    instruction=prompt.CRITIC_PROMPT,
    after_model_callback=_render_reference,
)
```

## Architecture

The web search agent is built on **Google ADK's SequentialAgent** architecture with two specialized sub-agents, enhanced with LangDB integration via langdb-adk:

1. **Critic Agent** (Google ADK LlmAgent with LangDB MCP integration)
   - Analyzes user queries to understand information needs
   - Conducts comprehensive web searches using LangDB's MCP server integration
   - Evaluates source reliability and information quality
   - Organizes findings and identifies information gaps

2. **Reviser Agent** (Google ADK LlmAgent with LangDB integration)
   - Receives research findings from the Critic Agent
   - Synthesizes information into coherent, comprehensive answers
   - Structures responses with appropriate formatting and organization
   - Ensures final answers directly address user queries

Both agents use Google ADK's threading system with a shared thread ID to maintain context throughout the process.

## Key Features

### 1. Sequential Processing
- Two-step approach ensures thorough research followed by quality synthesis
- Shared thread ID maintains context between agents

### 2. Comprehensive Web Search
- Multiple search strategies and varied search terms
- Focus on authoritative and recent sources
- Analysis of conflicting information and source reliability

### 3. Direct MCP Server Integration
- Web search capabilities through Tavily Search MCP
- Direct MCP server configuration in LangDBLlm
- Simplified setup without additional toolsets

### 4. Professional Synthesis
- Well-structured, comprehensive final answers
- Acknowledgment of limitations and conflicting information
- Clear, accessible writing style with proper organization

### 5. Model Flexibility
- Google ADK foundation with LangDB ADK extension supports multiple model providers
- Configurable model selection per agent via LangDB integration
- Direct MCP server integration through LangDB for enhanced capabilities

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
- LangDB ADK (extension of Google ADK for LangDB integration)

### Installation

```bash
pip install google-adk langdb-adk
```

### Main Dependencies

- `google-adk`: Google's Agents Development Kit for multi-agent systems
- `langdb-adk>=0.1.8`: Extension of Google ADK that adds LangDB integration with unified LLM interface and direct MCP server integration

## Model Configuration Details

### Current Configuration

Both agents are configured using Google ADK's LlmAgent with LangDBLlm (from langdb-adk extension) to connect to LangDB:

```python
# Critic Agent (with web search tools)
model=LangDBLlm(
    model="openai/gpt-4.1",
    api_key=os.getenv("LANGDB_API_KEY"),
    project_id=os.getenv("LANGDB_PROJECT_ID"),
    mcp_servers=[{
        "server_url": "https://api.us-east-1.langdb.ai/mcp/tavily_4ykdv5fj",
        "type": "sse",
        "name": "Tavily"
    }]
)

# Reviser Agent (synthesis only)
model=LangDBLlm(
    model="openai/gpt-4.1",
    api_key=os.getenv("LANGDB_API_KEY"),
    project_id=os.getenv("LANGDB_PROJECT_ID")
)
```

### Supported Model Formats

- OpenAI models: `"openai/gpt-4.1"`, `"openai/gpt-3.5-turbo"`
- Other Provider Models: Follow LangDB format `"anthropic/claude-sonnet-4"`
- LangDB Virtual Models: `"langdb/your-model-name"`

## References

* [Google ADK Documentation](https://google.github.io/adk-docs/)
* [LangDB Documentation](https://docs.langdb.ai/)
* [LangDB Virtual MCP Servers](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [LangDB Virtual Models](https://docs.langdb.ai/concepts/virtual-models)
* [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9)

---

Enjoy building comprehensive web search workflows with Google ADK + LangDB ADK! Configure your MCP servers directly in your agents or through virtual models in the LangDB dashboard for optimal search capabilities.