# Building Reliable AI Applications: LLMs, Evaluation, and Docker

*October 18, 2026*

In my previous article, I documented my first steps with Python, FastAPI, Pydantic, and SQLite.

Since then, I've taken three important steps toward becoming an AI engineer:

1. Integrating a local LLM into a REST API.
2. Evaluating AI-generated reports against trusted data.
3. Packaging an application using Docker.

Here are the most important lessons I've learned.

---

## 1. Integrating Qwen with FastAPI

### The objective

I wanted to create an API that retrieves solar generation data and asks Qwen, a locally running language model, to explain production deviations.

The workflow looks like this:

**HTTP Request → FastAPI → Python Calculations → Qwen → JSON Response**

I used Ollama to run Qwen locally.

### Calling an LLM with Python

```python
from ollama import chat

response = chat(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "system",
            "content": "You are a cautious energy analyst."
        },
        {
            "role": "user",
            "content": "Explain a 24% solar generation shortfall."
        }
    ],
    think=False
)

print(response.message.content)
```

An important concept is the distinction between system and user messages:

- **System message:** Defines how the model should behave.
- **User message:** Provides the task and relevant information.

### Mistake #1: Giving Qwen insufficient information

Initially, I provided only the plant name and date.

However, the model needed the actual generation, expected generation, and calculated shortfall.

I learned to construct a prompt containing the relevant data:

```python
prompt = f"""
Plant: {record["plant"]}
Date: {record["date"]}
Actual generation: {record["actual_mwh"]} MWh
Expected generation: {record["expected_mwh"]} MWh
Shortfall: {shortfall}%

Explain the deviation without inventing its cause.
"""
```

### Mistake #2: Incorrectly constructing the prompt

One of my tests failed because the model wasn't receiving the expected production numbers.

The problem was related to how I constructed and passed the prompt.

This reinforced the importance of Python f-strings, which allow variables to be inserted into text dynamically.

### Key takeaway

**Use conventional Python code for deterministic operations and LLMs for language-related tasks.**

Python should calculate the production shortfall. Qwen should explain the results.

Also, a production shortfall alone doesn't reveal its cause. The AI must avoid inventing weather conditions, equipment failures, or other explanations without evidence.

---

## 2. Evaluating AI-Generated Reports

### The objective

How can I trust an AI-generated report?

Imagine Qwen produces:

> Caucaia generated 76 MWh against an expected 100 MWh, representing a 30% shortfall.

The explanation sounds reasonable, but the correct shortfall is 24%.

I learned how to build a Python function to detect this kind of error.

### Separating AI output from trusted information

The evaluation function receives two dictionaries:

```python
report = {
    "plant": "Caucaia",
    "actual_mwh": 76.0,
    "expected_mwh": 100.0,
    "shortfall_pct": 30.0,
    "explanation": "Production was below expectations."
}

reference = {
    "plant": "Caucaia",
    "actual_mwh": 76.0,
    "expected_mwh": 100.0
}
```

The first contains AI-generated claims.

The second contains trusted reference data.

**The fundamental rule: Never use the model's own claims as the source of truth when verifying those claims.**

### Mistake #3: Trying to evaluate everything with another LLM

My initial instinct was to ask Qwen to generate an explanation and then extract percentages from the text.

I experimented with regular expressions.

However, the exercise was actually about validating structured data.

A much simpler approach was to compare dictionary values directly.

```python
correct_pct = calculate_shortfall_pct(
    reference["actual_mwh"],
    reference["expected_mwh"]
)

reported_pct = report["shortfall_pct"]

if abs(reported_pct - correct_pct) > 0.25:
    errors.append("Incorrect shortfall percentage.")
```

Here, `abs()` calculates the absolute difference.

The `0.25` represents the allowed tolerance in percentage points.

### Mistake #4: Assuming all fields exist

Initially, I accessed dictionary values directly:

```python
reported_pct = report["shortfall_pct"]
```

But what happens if the AI forgets to include that field?

Python raises a `KeyError`.

A safer approach uses `.get()`:

```python
reported_pct = report.get("shortfall_pct")

if reported_pct is None:
    errors.append("Shortfall percentage is missing.")
```

### Mistake #5: Confusing strings and numbers

I initially tried to validate production values as strings.

However, production measurements should be numeric:

```python
if not isinstance(actual, (int, float)):
    errors.append("Actual generation must be a number.")
```

I also learned how to validate explanations:

```python
if not isinstance(explanation, str) or not explanation.strip():
    errors.append("Explanation must be a nonempty string.")
```

The `.strip()` method removes whitespace from both ends of a string, helping detect explanations containing only spaces.

### Returning evaluation results

My function eventually returned a dictionary like:

```python
{
    "passed": False,
    "errors": [
        "Incorrect shortfall: expected 24.0%, received 30.0%."
    ]
}
```

The expression:

```python
len(errors) == 0
```

returns `True` only when no validation errors were found.

### Key takeaway

**AI evaluation doesn't always require AI.**

Many errors can be detected using deterministic Python logic.

This doesn't guarantee that an entire AI-generated explanation is correct, but it provides an important reliability layer.

---

## 3. Running My First Application with Docker

### The objective

Until now, my applications depended on Python packages installed on my MacBook.

I wanted to understand how to package an application so it could run in an isolated environment.

That's where Docker comes in.

Docker packages an application and its dependencies into an image that can be executed as a container.

### Understanding a Dockerfile

The five instructions I learned were:

| Instruction | Purpose |
|-------------|---------|
| `FROM` | Select a base image |
| `WORKDIR` | Set the working directory |
| `COPY` | Copy application files |
| `RUN` | Execute commands during image creation |
| `CMD` | Define the startup command |

### Mistake #6: Incorrectly starting Uvicorn

My first Docker container failed because I incorrectly specified the application startup command.

Uvicorn expects:

```text
module:application
```

For my FastAPI application:

```dockerfile
CMD ["python", "-m", "uvicorn", "exercise_07_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

The `:app` refers to the FastAPI application object:

```python
app = FastAPI()
```

I had encountered this problem before when running FastAPI outside Docker!

### Mistake #7: Missing dependencies inside Docker

Another error was:

```text
No module named uvicorn
```

Initially, I wondered why this happened since Uvicorn was already installed in my Python virtual environment.

Then I understood:

**My MacBook's Python environment and the Docker container's Python environment are independent.**

The container needs its own dependencies.

I created a requirements file containing packages such as:

```text
fastapi
uvicorn
pydantic
ollama
```

The Dockerfile installs them using:

```dockerfile
RUN pip install --no-cache-dir -r requirements-docker.txt
```

### Building and running the container

After fixing the Dockerfile, I successfully executed:

```bash
docker build -t solar-ai-lab .
```

And:

```bash
docker run --rm -p 8000:8000 solar-ai-lab
```

The application became accessible at:

```text
http://localhost:8000/docs
```

The `-p 8000:8000` argument maps port 8000 on my Mac to port 8000 inside the container.

### Key takeaway

**Docker makes applications more portable by packaging their runtime dependencies.**

However, packaging an application is not the same as deploying a complete production system. Configuration, security, monitoring, and external services still need attention.

---

## 4. What I've Learned About AI Engineering

These exercises changed how I think about building AI applications.

Initially, I focused primarily on interacting with language models.

Now I understand that a functional AI application needs much more:

- Reliable data retrieval.
- Deterministic calculations.
- Input and output validation.
- APIs for communication.
- Automated testing.
- AI evaluation.
- Reproducible execution environments.

An LLM is just one component of the system.

**My biggest lesson so far: AI engineering is as much about building reliable software as it is about understanding AI models.**

---

## 5. What's Next?

My next project will combine everything I've learned into a single application:

### Solar Plant AI Analyst

The architecture will be:

**SQLite → FastAPI → Python → Qwen → Report Evaluation → Docker**

My immediate next challenge is connecting a Dockerized FastAPI application to Ollama running on my MacBook.

Eventually, I want to build an application capable of retrieving actual energy production data, identifying performance deviations, and generating reliable reports for human review.

I'll continue documenting the mistakes, lessons, and technical decisions along the way.