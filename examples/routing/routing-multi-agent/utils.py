from openai import OpenAI
import uuid
from typing import Dict, Any, List
import requests
import os
import json
from dotenv import load_dotenv


load_dotenv()
base_url = "https://api.us-east-1.langdb.ai"


def tag_headers(project_id, tags=None):
    return {
        "x-project-id": project_id,
        "x-thread-id": str(uuid.uuid4()),
        "x-tags": tags,
    }


def completion(
    client: OpenAI,
    model,
    messages: List[Dict[str, str]],
    headers: Any = None,
    extra_body: Any = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
):
    response = client.chat.completions.create(
        model=model,  # Use the model
        messages=messages,  # Define the interaction
        temperature=temperature,  # Control the creativity of the response
        max_tokens=max_tokens,  # Limit the length of the response
        extra_headers=headers,
        extra_body=extra_body,
    )
    return response.choices[0].message.content.strip()


def get_traces(thread_id: str, project_id: str) -> dict:
    """
    Master function that returns all the traces for a thread.
    """
    base_url = "https://api.us-east-1.langdb.ai"
    url = base_url + "/query"
    query = f"""
    WITH table_spans AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY trace_id ORDER BY start_time_us ASC) AS rn
      FROM langdb.traces
      WHERE operation_name = 'model_call' 
    )
    SELECT 
      t.trace_id,
      t.span_id,
      t.parent_span_id,
      t.operation_name,
      t.start_time_us,
      t.finish_time_us,
      t.attribute,
      t.thread_id,
      child.attribute as child_attribute, 
      child.span_id as child_span_id
    FROM langdb.traces AS t
    LEFT JOIN table_spans AS child
    ON t.trace_id = child.trace_id AND child.rn = 1
    WHERE t.parent_span_id = '0'
    AND NOT(ISNULL(child.span_id)) AND child.span_id <> '0' AND t.thread_id = '{thread_id}'
    ORDER BY start_time_us DESC
    LIMIT 100;
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LANGDB_STAGING_KEY')}",
        "x-project-id": project_id,
    }

    payload = {"query": query}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


def get_costs_and_durations(thread_id: str, project_id: str) -> list[dict]:
    """
    Returns a list of dictionaries, each containing:
    {
      "cost": float,
      "duration": float,
      "trace_id": str,
      "span_id": str
    }
    """
    data = get_traces(thread_id, project_id).get("data", [])
    results = []

    for row in data:
        # ---- Duration Calculation ----
        start_str = row.get("start_time_us", "0")
        finish_str = row.get("finish_time_us", "0")
        try:
            start = float(start_str)
            finish = float(finish_str)
            duration = finish - start
        except (ValueError, TypeError):
            duration = 0.0

        # ---- Cost Calculation ----
        cost_val = 0.0
        child_attr = row.get("attribute", {})
        cost_val_raw = child_attr.get("cost") if child_attr.get("cost") else 0
        cost_val = parse_cost(cost_val_raw)

        # ---- Collect results ----
        results.append(
            {
                "trace_id": row.get("trace_id", ""),
                "span_id": row.get("span_id", ""),
                "cost": cost_val,
                "duration": duration,
            }
        )

    return results


def query_traces(tag: str, project_id: str) -> list[dict]:
    """
    Returns a list of dictionaries, each containing:
    {
      "cost": float,
      "duration": float,
      "trace_id": str,
      "span_id": str
    }
    """
    base_url = "https://api.staging.langdb.ai"
    url = base_url + "/query"

    query = f"""
    WITH table_spans AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY trace_id ORDER BY start_time_us ASC) AS rn
      FROM langdb.traces
      WHERE operation_name = 'model_call' 
    )
    SELECT DISTINCT
      t.thread_id
    FROM langdb.traces AS t
    LEFT JOIN table_spans AS child
    ON t.trace_id = child.trace_id AND child.rn = 1
    WHERE t.parent_span_id = '0'
    AND NOT(ISNULL(child.span_id)) AND child.span_id <> '0' AND t.tags['exp'] = '{tag}'
    ORDER BY start_time_us ASC
    LIMIT 100;
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LANGDB_STAGING_KEY')}",
        "x-project-id": project_id,
    }

    payload = {"query": query}
    response = requests.post(url, headers=headers, json=payload)

    thread_id_rows = response.json().get("data", [])
    thread_ids = [row["thread_id"] for row in thread_id_rows]
    return thread_ids


def parse_cost(cost_val):
    """
    cost_val might be:
    - A JSON string with additional info (e.g. '{"cost":0.010209,"per_input_token":3.0,"per_output_token":12.0}')
    - A float or int directly
    - None or empty
    """
    if cost_val is None:
        return 0.0

    if isinstance(cost_val, (float, int)):
        # Already numeric
        return float(cost_val)

    if isinstance(cost_val, str):
        # Likely a JSON string
        try:
            cost_dict = json.loads(cost_val)
            # The actual cost is inside the key "cost"
            return float(cost_dict.get("cost", 0.0))
        except json.JSONDecodeError:
            # If it fails to parse, treat it as 0.0
            return 0.0

    # Fallback
    return 0.0
