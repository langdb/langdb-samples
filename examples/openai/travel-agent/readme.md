# OpenAI Agents SDK Example: Multi‑Agent Travel Query Workflow

This example demonstrates how to build a multi-agent travel query workflow using the OpenAI Agents SDK, with guardrails and tools attached to your agents via LangDB virtual models (MCP servers).

---

##  Prerequisites

1. **API Keys & Endpoints**

   | Variable            | Description                                                                   |
   | ------------------- | ----------------------------------------------------------------------------- |
   | `LANGDB_API_KEY`    | Your LangDB token for managing virtual models and MCP servers.                |
   | `LANGDB_BASE_URL`   | Base URL of LangDB API (e.g., `https://api.us-east-1.langdb.ai`).             |
   | `LANGDB_PROJECT_ID` | Your LangDB project identifier (used to namespace API calls in your account). |

   ```bash
   export LANGDB_API_KEY="ld-..."
   export LANGDB_BASE_URL="https://api.us-east-1.langdb.ai"
   export LANGDB_PROJECT_ID="your-project-id"
   ```

2. **Install Dependencies**

   ```bash
   pip install openai openai-agents python-dotenv
   ```

---

# Model Configuration

Models are configured individually for each agent in `main.py`. Each agent uses the `create_llm()` function to configure its language model.

> We need to setup Virtual MCP Server and Virtual Model for `Travel Recommendation Specialist` to use Tavily Search MCP.
> Make sure you configure these guardrails under Guardrails, Virtual MCP Servers under MCP Server in the LangDB dashboard before selecting them in your Virtual Model setup.
## Virtual MCP Server Setup

1. Log in and navigate to **MCP Servers → Virtual MCP Servers** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual MCP Server** and configure:
   - **Name**: e.g. `web-search-mcp`  
   - **Underlying MCP**: choose [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9)
3. Then head to Models page to setup virtual model.

## Virtual Model Setup

1. Log in and navigate to **Project → Models** on [app.langdb.ai](https://app.langdb.ai).  
2. Click **+ New Virtual Model** and configure name, base model, version, and MCP Server for search tools. We used [Tavily Search MCP](https://app.langdb.ai/mcp-servers/tavily-mcp-4024f9c3-3d20-48d2-92da-4c7e9910e5f9) Deployed on LangDB. 
3. Copy the generated model name (e.g. `openai/langdb/your-model-name`).  
4. Update your `main.py` for `Travel Recommendation Specialist` pass that virtual-model name into `model=get_model(....)`, for example:

```python
travel_recommendation_agent = Agent(
    name="Travel Recommendation Specialist",
    model=get_model("langdb/recc_8ac7wclb", client),  # langDB virtual model here
    model_settings=ModelSettings(
        tool_choice='auto',
    ),
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a travel recommendation specialist. You help customers find ideal destinations and travel plans. Use the duckduckgo to find the best options.",
)

```

## Architecture


1. **Query Router Agent** (virtual model with guardrails)
2. **Booking Specialist** (virtual model)
3. **Travel Recommendation Specialist** (virtual model + search tool using Virtual MCP Server)
4. **Reply Agent** (model for emoji‑rich formatting)

Agents are wired as tools and handoffs via MCP servers—the router calls booking or travel tools, then hands off to reply for final formatting.

### Guardrails

#### Query Router Agent

| Guardrail | Purpose | Typical Threshold |
|-----------|---------|-------------------|
| **Topic Adherence (Travel)** | Rejects queries that are not about travel. | Rejection if similarity < 0.4 |
| **OpenAI Moderation** | Blocks hate, self‑harm, violence, sexual, or other disallowed content as per OpenAI policy. | Default category thresholds |
| **Minimum Word Count** | Ensures requests contain enough context for quality answers. | Reject if < 8 words |

## Key Features

### 1. Agent‑as‑Tool Pattern

* Each specialist is exposed as a tool on the Query Router virtual model.
* Tools are invoked automatically based on the router’s decision.

### 2. Virtual Model‑based Guardrails

* All input/output validation is handled by the virtual model (MCP server) config.
* No need to implement guardrails in code—just define them in LangDB.

### 3. Tool Integration via MCP

* Attach external tools (like web search) directly to your virtual model.
* 🔍 The Travel Recommendation model has a DuckDuckGo search tool enabled.

### 4. Real‑time Streaming & Debugging

* Stream tool calls, response deltas, and agent handoffs in the console.

---

##  Running the Example

```bash
python run.py "I'm planning a trip to Japan in April. What are the must‑see cherry blossom spots?"
```

* Queries are first routed by the **Query Router Agent**.
* Tools (booking or travel) run after the router's decision.
* Final output is formatted by the **Reply Agent** (adds emojis).

---

## 📚 References

* [LangDB Virtual MCP Servers](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [LangDB Virtual MCP](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [LangDB Guardrails](https://docs.langdb.ai/features/guardrails)
* [OpenAI Agents SDK](https://github.com/openai/agents-sdk)
---

Enjoy building multi-agent workflows with LangDB + OpenAI Agents SDK! Feel free to tweak guardrails and tools via your Virtual Model configurations.
