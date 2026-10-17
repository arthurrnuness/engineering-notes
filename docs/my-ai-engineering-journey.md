# My AI Engineering Journey: From Python Scripts to APIs

*October 17, 2026*

I'm an electrical engineer with a background in software development and energy operations.

Recently, I decided to explore AI engineering, particularly how to build applications using large language models (LLMs), Python, and AI agents.

This article documents what I've learned so far, the mistakes I've made, and how I solved them.

## 1. Running an LLM Locally

My first experiment involved running Qwen locally using Ollama.

I learned that an LLM is not the same thing as an AI agent.

- **Qwen:** The language model that generates responses.
- **Ollama:** The software that runs the model locally.
- **LangChain:** A framework for connecting models to tools and coordinating workflows.
- **Python:** The language used to build the application.

One important discovery was that an LLM doesn't automatically have access to external information.

If I want it to retrieve energy generation data, I must provide a tool that performs that operation.

## 2. Understanding AI Agents

I built a simple agent capable of retrieving generation data and calculating production shortfalls.

The workflow was:

1. Receive a question from the user.
2. Let Qwen decide which tool to use.
3. Execute the selected Python function.
4. Return the result to Qwen.
5. Generate a response based on the retrieved data.

I also learned why agents need loops.

When an LLM requests a tool, the application must execute it and return the result. The model may then request another tool before producing its final answer.

**Key takeaway:** An AI agent is more than an LLM. It combines a model, tools, and an execution loop.

## 3. Python: Dictionaries and Unpacking

One concept I struggled with was dictionary unpacking.

Consider:

```python
record = {
    "plant": "Caucaia",
    "actual_mwh": 76.0
}
```

To add another field without modifying the original dictionary:

```python
result = {
    **record,
    "shortfall_pct": 24.0
}
```

The `**` operator unpacks the existing dictionary into the new one.

This is equivalent to:

```python
result = record.copy()
result["shortfall_pct"] = 24.0
```

Another useful distinction:

- Lists use `.append()` to add elements.
- Dictionaries use `dictionary["key"] = value`.

## 4. Building My First API with FastAPI

I learned how to expose Python functions through HTTP endpoints.

My first API included:

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

I also created an endpoint that retrieves solar generation data and calculates production shortfalls.

### Mistake #1: Incorrect Uvicorn Command

I initially tried:

```bash
python -m uvicorn exercise_07_fastapi.py --reload
```

This failed because Uvicorn expects a module name and an application object.

The correct command was:

```bash
python -m uvicorn exercise_07_fastapi:app --reload
```

### Mistake #2: Returning the Wrong JSON Structure

I initially returned a list containing the record and its calculated shortfall.

However, the API expected a single dictionary with an additional field.

The solution:

```python
return {**result, "shortfall_pct": shortfall}
```

### Mistake #3: Incorrect Field Names

My automated tests failed because I returned:

```python
"shortfall": 24.0
```

But the API contract expected:

```python
"shortfall_pct": 24.0
```

**Lesson learned:** Correct calculations aren't enough. APIs must also return the expected data structure and field names.

## 5. Data Validation with Pydantic

Next, I learned how Pydantic validates incoming data.

```python
class GenerationInput(BaseModel):
    plant: str = Field(min_length=1)
    actual_mwh: float = Field(ge=0)
    expected_mwh: float = Field(gt=0)
```

This prevents invalid values from reaching my application logic.

For example, actual generation cannot be negative, and expected generation must be greater than zero.

### Mistake #4: Confusing Percentages and Decimals

My application needed to classify plants as underperforming when their shortfall reached 10%.

I initially compared the result against `0.1`.

But my calculation returned percentages such as `24.0`, not fractions such as `0.24`.

The correct condition was:

```python
if shortfall_pct >= 10:
    status = "underperforming"
else:
    status = "normal"
```

**Lesson learned:** Always understand the units and conventions used by a function.

## 6. Working with SQLite

I also learned how to store solar generation data in a relational database.

My first task was importing CSV records into SQLite.

### Mistake #5: Building SQL Queries with F-Strings

I initially tried inserting values directly into an SQL statement.

I learned that parameterized queries are safer and easier to maintain.

```python
conn.execute(
    """
    INSERT INTO generation
    (plant, date, actual_mwh, expected_mwh)
    VALUES (?, ?, ?, ?)
    """,
    (
        row["plant"],
        row["date"],
        float(row["actual_mwh"]),
        float(row["expected_mwh"])
    )
)
```

The `?` placeholders separate SQL instructions from the values supplied by Python.

### Mistake #6: Passing a String Instead of a Tuple

When querying a specific plant, I initially passed the plant name directly:

```python
conn.execute(query, plant)
```

The correct approach was:

```python
conn.execute(query, (plant,))
```

The comma creates a single-element tuple.

### Discovering sqlite3.Row

I also learned that SQLite normally returns rows as tuples.

By setting:

```python
conn.row_factory = sqlite3.Row
```

I can access columns by name:

```python
row["plant"]
row["actual_mwh"]
```

This makes the code more readable.

## 7. Learning to Test My Code

One of my most important discoveries was automated testing.

Using pytest, I could verify that my code behaved as expected.

I experienced failing tests caused by:

- Incorrect field names.
- Incorrect conditional logic.
- Unexpected return structures.

Rather than treating these failures as setbacks, I started using them to understand what my application was supposed to do.

**Key takeaway:** A program running successfully doesn't necessarily mean it's behaving correctly.

## What's Next?

My goal is to become proficient in applied AI engineering.

My next learning objectives are:

1. Integrating LLMs into FastAPI applications.
2. Building reliable AI agents connected to databases.
3. Learning Docker and cloud deployment.
4. Understanding AI evaluation and monitoring.
5. Building a production-ready Solar Plant AI Analyst.

The long-term objective is to connect software engineering, artificial intelligence, and my experience in the energy industry.

I'll continue documenting what works, what fails, and what I learn along the way.