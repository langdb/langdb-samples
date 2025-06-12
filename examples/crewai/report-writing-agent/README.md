# CrewAI Report Writing Agent

A multi-agent report generation system built with CrewAI that creates comprehensive research reports on any given topic.

## Overview

This application uses three specialized AI agents working in sequence:
- **Researcher**: Gathers comprehensive information on the topic
- **Analyst**: Analyzes and synthesizes the research findings
- **Report Writer**: Creates a professional Markdown report

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

Models are configured individually for each agent in `main.py`. Each agent uses the `create_llm()` function to configure its language model.

> We need to setup Virtual MCP Server and Virtual Model for `Researcher Agent` to use Tavily Search MCP.

## Virtual MCP Server Setup

1. Log in and navigate to **MCP Servers → Virtual MCP Servers** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual MCP Server** and configure:
   - **Name**: e.g. `web-search-mcp`  
   - **Underlying MCP**: choose [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9)
3. Then head to Models page to setup virtual model.

## Virtual Model Setup

1. Log in and navigate to **Project → Models** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual Model** and configure name, base model, version, and MCP Server for search tools. We used [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9) Deployed on LangDB. 
3. Copy the generated model name (e.g. `openai/langdb/report-researcher@v1`).  
4. Update your `main.py` for `Researcher Agent` pass that virtual-model name into `create_llm(...)`, for example:

```python
      @agent
      def researcher(self) -> Agent:
         return Agent(
               config=self.agents_config['researcher'],
               verbose=True,
               llm=create_llm("openai/langdb/your-model-name", "research")
         )
```

### Code

Models are configured individually for each agent in main.py. Each agent uses the create_llm() function to configure its language model.

```python

llm=create_llm("openai/anthropic/claude-3-5-sonnet-20240620", "research")

llm=create_llm("openai/gpt-4.1", "analysis")


llm=create_llm("openai/gemini/gemini-2.5-pro-preview", "report_writer")
```

### Changing Models

To change the model for any agent, modify the model string in the corresponding agent method:

1. **Researcher Agent**: Edit line 49 in `main.py`
2. **Analyst Agent**: Edit line 57 in `main.py`
3. **Report Writer Agent**: Edit line 65 in `main.py`

### Supported Model Formats


- OpenAI models: `"openai/gpt-4.1"`, `"openai/gpt-3.5-turbo"`
- Other Provider Models are in format of LiteLLM so `openai/anthropic/claude-sonnet-4` 



## Usage

### Command Line

```bash
# Run with topic as argument
python main.py "Your research topic here"

# Run interactively (will prompt for topic)
python main.py
```

### Example

```bash
python main.py "The Impact of Artificial Intelligence on Social Media Marketing: A 2024 Report"
```

## Output

The application generates:
- Console output showing the progress of each agent
- A `report.md` file containing the final report
- Preview of the report in the terminal

## Project Structure

```
main.py                 # Main application entry point
configs/
   agents.yaml        # Agent configurations (roles, goals, backstories)
   tasks.yaml         # Task configurations and workflows
pyproject.toml         # Project dependencies and metadata
README.md              # This file
```

## Agent Configuration

Agents are configured in `configs/agents.yaml`:
- **researcher**: Domain expert focused on gathering current information
- **analyst**: Data specialist for synthesizing insights
- **report_writer**: Technical writer for creating structured reports

## Task Configuration

Tasks are defined in `configs/tasks.yaml`:
- **research_task**: Comprehensive topic research
- **analysis_task**: Data analysis and insight extraction
- **report_writing_task**: Professional report creation

## Dependencies

### Prerequisites

- Python 3.8+
- [UV](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installation

1. First, install UV (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Follow the on-screen instructions to add UV to your PATH.

2. Install project dependencies using UV:
   ```bash
   uv pip install -e .
   ```
   
   This will install all the required packages listed in `pyproject.toml`.

### Main Dependencies

- `crewai>=0.108.0`: Multi-agent orchestration framework
- `crewai-tools>=0.0.1`: Additional tools for CrewAI
- `python-dotenv>=1.0.1`: Environment variable management


## References

* [LangDB Virtual MCP Servers](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [LangDB Virtual MCP](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [CrewAI Documentation](https://docs.crewai.com/)