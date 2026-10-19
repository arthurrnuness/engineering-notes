# From APIs to AI Agents: Tool Calling and Reliable Answers

*October 19, 2026*

In my previous note, I covered integrating Qwen into a FastAPI application, evaluating reports, and running the application with Docker. Since then, I've been turning my **Solar Plant AI Analyst** into a tool-calling agent that can retrieve data, perform calculations, and explain what it finds.

## 1. Bringing the application together

I connected the pieces into `solar_app.py`: FastAPI endpoints, a SQLite database, Python calculations, and Qwen running locally through Ollama. I also added a `POST /generation` endpoint to save new records and used `pytest`, `TestClient`, and `unittest.mock.patch` to test API behavior without relying on a real model response every time.

When running the API in Docker, I learned two practical lessons:

- The container can reach Ollama on my Mac through `host.docker.internal:11434`.
- SQLite data needs a mounted volume if I want changes to persist beyond a container's lifetime.

The application works locally; I haven't deployed it as a public production service.

## 2. Structured output is not the same as correct output

I experimented with Pydantic and Ollama's structured-output feature:

```python
from typing import Literal
from pydantic import BaseModel
from ollama import chat

class AnalysisOutput(BaseModel):
    severity: Literal["low", "medium", "high"]
    summary: str
    investigation_required: bool

response = chat(
    model="qwen3.5:9b",
    messages=messages,
    format=AnalysisOutput.model_json_schema(),
    think=False,
)

analysis = AnalysisOutput.model_validate_json(response.message.content)
```

Pydantic helps ensure that the result has the expected **shape and types**. But Qwen classified a **24% generation shortfall as `medium`** when my application rules required `high`.

My solution was to move severity thresholds into Python. **The model can explain a classification, but deterministic business rules should calculate it.**

## 3. Giving Qwen more than one tool

I created two functions for the agent:

- `find_record_db(plant, date)` retrieves one day's generation.
- `get_plant_history(plant, limit=7)` retrieves recent records.

I passed both functions to Qwen. Depending on the question, it requested one or both. I used a **tool registry** instead of hardcoding a separate execution branch for every function:

```python
available_tools = {
    "find_record_db": find_record_db,
    "get_plant_history": get_plant_history,
}

function = available_tools.get(tool_call.function.name)
if function is None:
    raise ValueError("Unknown tool requested")

result = function(**tool_call.function.arguments)
```

The key distinction: **Qwen requests the function; Python executes it.** The registry also limits execution to functions I explicitly allow.

## 4. Understanding conversation history and agent loops

I initially wondered why I needed to append messages after a tool call. Now I understand that the model doesn't automatically receive the result of Python's execution.

A typical exchange is:

```text
USER       → Analyze Caucaia's recent generation.
ASSISTANT  → Request get_plant_history(...).
TOOL       → Return the SQLite records.
ASSISTANT  → Explain the results, or request another tool.
```

The application must preserve the assistant's tool request and append the tool's result before calling Qwen again.

I also learned why agents need **two kinds of loops**:

- A `for` loop executes all tool calls requested **in one model response**.
- An outer loop lets the model inspect the results and request **more tools in another round**.

I replaced `while True` with a bounded loop (`MAX_ROUNDS = 5`) so the agent cannot request tools indefinitely. I also added tracing:

```text
Agent round: 1
Executing tool: get_plant_history
Executing tool: find_record_db
Agent round: 2
Final answer
```

Seeing the tool names and arguments made it easier to understand what the agent was actually doing. In one run, Qwen requested **30 or 50 records even though only three existed**—a reminder that tool arguments need validation and sensible limits.

## 5. Giving the model verified facts

My database contains three sample generation records for Caucaia:

| Date | Actual (MWh) | Expected (MWh) | Verified performance |
|---|---:|---:|---|
| 2026-10-14 | 76 | 100 | Below expected |
| 2026-10-15 | 89 | 95 | Below expected |
| 2026-10-16 | 100 | 98 | Above expected |

Qwen once described the generation as **declining**, even though it increased from October 14 to 16. The database returned records in reverse chronological order, and the model interpreted the sequence backward.

So I created Python functions to calculate the facts first:

```python
def calculate_trends(records):
    if len(records) < 2:
        return "insufficient_data"

    chronological = sorted(records, key=lambda row: row["date"])
    change = chronological[-1]["actual_mwh"] - chronological[0]["actual_mwh"]

    if change > 0:
        return "increasing"
    if change < 0:
        return "decreasing"
    return "stable"


def classify_performance(record):
    actual = record["actual_mwh"]
    expected = record["expected_mwh"]

    if actual > expected:
        return "above_expected"
    if actual < expected:
        return "below_expected"
    return "on_target"
```

The trend calculation compares the **first and last available records**; it doesn't prove every intermediate value follows the same direction.

I then returned the original records **plus** `verified_trend` and `verified_performance` in the tool result. Qwen subsequently reported the increasing trend and all three daily classifications correctly.

## 6. What still goes wrong

Even with accurate tool results, I saw Qwen:

- Invent a wider date range than the database contained.
- Say October 15 met expectations when it was actually below target.
- Confuse **MW** (power) with **MWh** (energy).
- Suggest operational causes or an efficiency review without supporting data.
- Call a small surplus "significant" without a defined threshold.

These errors showed me that **tool calling, grounding, and factual correctness are separate problems**. A model may retrieve the correct numbers and still write an inaccurate explanation.

## What's next?

My next exercise is to make the agent's final claims structured with Pydantic, then compare its reported trend and daily classifications against the facts computed by Python. I want the application to **detect contradictions automatically**, rather than depending on me to spot them by reading the response.

**Main takeaway:** I'm learning that an AI agent is not just a prompt connected to functions. It's a software system that needs controlled tools, explicit conversation state, bounded execution, traceability, and independent checks of what the model says.
