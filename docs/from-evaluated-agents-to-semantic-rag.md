# From Evaluated AI Agents to Semantic RAG: Building Systems I Can Actually Test

*October 20, 2026*

In my [previous note](from-apis-to-ai-agents.md), I described how I turned a solar generation API into a tool-calling AI agent. I had connected a local Qwen model to Python functions and SQLite, learned to maintain the conversation history, and built a bounded agent loop. More importantly, I had discovered that **a model can retrieve the correct data and still explain it incorrectly**.

Since then, I have worked through **Exercises 23–45**. This phase had two main parts:

1. Making my AI agent's conclusions **structured, independently evaluated, and gated before display**.
2. Building a **retrieval-augmented generation (RAG)** pipeline, moving from basic keyword matching to embeddings, semantic search, chunking, and automated retrieval tests.

The biggest change in how I think about AI engineering is that I no longer consider a plausible answer to be enough. I want to know **what evidence the system had, why it retrieved that evidence, what claims it made, and whether those claims passed a test**.

This entry is longer than my usual notes because I want to preserve the actual implementations, mistakes, numerical results, and lessons from the exercises. The examples come from my local learning project, not a production deployment.

---

## 1. Starting point: a working agent with a reliability problem

The project is my **Solar Plant AI Analyst**, running locally on my Mac. The relevant stack includes:

- **Python** for application logic and deterministic calculations.
- **SQLite** for sample solar generation data.
- **Ollama + `qwen3.5:9b`** for reasoning, tool selection, and natural-language explanations.
- **Pydantic** for structured responses.
- Later, **`nomic-embed-text`** for embeddings and semantic retrieval.

The agent already knew how to execute two Python tools:

```python
available_tools = {
    "get_plant_history": get_plant_history,
    "find_record_db": find_record_db,
}
```

It could ask for historical generation data, inspect a particular date, and summarize the findings. I also had a `MAX_ROUNDS = 5` limit to prevent unbounded model/tool loops.

My example database contained three Caucaia records:

| Date | Actual generation | Expected generation | Correct classification |
|---|---:|---:|---|
| 2026-10-14 | 76 MWh | 100 MWh | Below expected |
| 2026-10-15 | 89 MWh | 95 MWh | Below expected |
| 2026-10-16 | 100 MWh | 98 MWh | Above expected |

Python could correctly calculate that generation had increased from **76 to 100 MWh** across the available records. But Qwen had previously reversed the trend, confused units, invented broader date ranges, and made unsupported operational conclusions.

So the next question was: **How can I automatically detect when the model's answer disagrees with the data?**

---

## 2. Exercises 23–24: Structured claims and an independent trend evaluator

### 2.1 Defining an answer contract with Pydantic

I started by creating a Pydantic model describing the claims I wanted to verify:

```python
from pydantic import BaseModel
from typing import Literal


class AgentClaims(BaseModel):
    trend: Literal[
        "increasing",
        "decreasing",
        "stable",
        "insufficient_data",
    ]

    performance_by_date: dict[
        str,
        Literal["above_expected", "below_expected", "on_target"],
    ]

    summary: str
```

This gave me a machine-readable representation of the final answer:

```json
{
  "trend": "increasing",
  "performance_by_date": {
    "2026-10-14": "below_expected",
    "2026-10-15": "below_expected",
    "2026-10-16": "above_expected"
  },
  "summary": "Generation increased from October 14 to October 16."
}
```

Instead of trying to parse arbitrary prose to determine whether Qwen claimed a trend was increasing, I could read `claims.trend` directly.

There are three Pydantic methods I wanted to remember:

```python
AgentClaims.model_json_schema()
# Returns a JSON Schema describing the allowed output structure.

claims = AgentClaims.model_validate_json(json_text)
# Parses a JSON string and validates it into a Pydantic model.

claims.model_dump()
# Returns a Python dictionary containing the model's fields.
```

I also used `claims.model_dump_json()` when thinking about the difference between a Python dictionary and a JSON string.

**One mistake I made:** my first `AgentClaims` class included only `trend`. Qwen returned:

```python
{"trend": "increasing"}
```

Initially, I wondered why the other fields were missing. The model wasn't necessarily ignoring the schema: **I had accidentally defined an incomplete schema**.

These two debugging statements exposed the problem:

```python
print(AgentClaims.model_json_schema())
print(AgentClaims.model_fields.keys())
```

Once I added the missing fields, the schema correctly required `trend`, `performance_by_date`, and `summary`, and Qwen returned all three.

**Lesson:** Schema enforcement does not protect an application against an incorrectly designed schema. It enforces what the developer specified, including developer mistakes.

### 2.2 Requesting structured output from Ollama

After the tool-calling portion of the agent finished, I made a separate Ollama call asking for the result in the `AgentClaims` format:

```python
structured_response = chat(
    model="qwen3.5:9b",
    messages=messages + [{
        "role": "user",
        "content": (
            "Using the retrieved data, return the trend, "
            "performance_by_date, and a short summary. "
            "Include only dates present in the data."
        ),
    }],
    format=AgentClaims.model_json_schema(),
    think=False,
)

claims = AgentClaims.model_validate_json(
    structured_response.message.content
)
```

At this stage, the model still generated the claims. Pydantic only checked that the resulting values and types conformed to the declared structure.

**Structural validity and factual correctness are different tests.** A valid `"decreasing"` value could still be factually wrong for Caucaia.

### 2.3 Calculating ground truth separately

I kept objective calculations in Python rather than delegating them to Qwen:

```python
def calculate_trends(records):
    if len(records) < 2:
        return "insufficient_data"

    chronological = sorted(records, key=lambda row: row["date"])
    first = chronological[0]
    last = chronological[-1]

    change_mwh = last["actual_mwh"] - first["actual_mwh"]

    if change_mwh > 0:
        return "increasing"
    elif change_mwh < 0:
        return "decreasing"
    return "stable"


def classify_performance(record):
    deviation_mwh = record["actual_mwh"] - record["expected_mwh"]

    if deviation_mwh > 0:
        return "above_expected"
    elif deviation_mwh < 0:
        return "below_expected"
    return "on_target"
```

For Caucaia, this gives:

```python
verified_trend = "increasing"

verified_performance = {
    "2026-10-16": "above_expected",
    "2026-10-15": "below_expected",
    "2026-10-14": "below_expected",
}
```

An important nuance: my trend function compares the **earliest and latest available values**. It does *not* prove that each intermediate observation increases monotonically, establish a statistically significant trend, or explain why generation changed.

### 2.4 Testing a claim against ground truth

My first automatic check was simple:

```python
trend_passed = claims.trend == verified_trend
```

I initially forgot to update `verified_trend` after calculating `trend`. It remained `None`, so every comparison failed, regardless of the model's answer. That was a state-management bug, not a model-quality problem.

The fix was to preserve the value when the historical-data tool returned:

```python
trend = calculate_trends(records)
verified_trend = trend
```

I deliberately changed a valid claim to `"decreasing"` and verified that the evaluation failed. This was my first small **negative test**: I wasn't just checking that correct behavior passed; I was proving that an incorrect claim could be detected.

---

## 3. Exercises 25–26: Evaluating every date and rejecting failed answers

### 3.1 Checking a dictionary of daily classifications

I extended the evaluation from one trend to all three dates:

```python
verified_performance = {
    row["date"]: classify_performance(row)
    for row in records
}

performance_passed = (
    claims.performance_by_date == verified_performance
)
```

Python dictionaries compare by **keys and values, not insertion order**. This made it possible to detect not only wrong classifications, but also missing or unexpected dates.

I tested a deliberate mistake:

```python
claims.performance_by_date["2026-10-15"] = "above_expected"
```

The trend still passed, but the daily performance comparison failed. That is exactly what I wanted: **a narrowly defined check that can detect a specific error**.

### 3.2 Detecting errors is not the same as blocking them

My next mistake was subtler. I had code that calculated the evaluation results and then printed the structured answer **before** checking whether the evaluation passed:

```python
trend_passed = claims.trend == verified_trend
performance_passed = claims.performance_by_date == verified_performance

print(claims.model_dump())  # Oops: always displayed.

if trend_passed and performance_passed:
    print("EVALUATION PASSED")
else:
    print("EVALUATION FAILED")
```

I asked why the print was considered premature, since it appeared *after* the comparisons. I realized the distinction:

- **Evaluation computation:** Calculate whether the result is valid.
- **Output gating:** Use that result to decide whether the answer is allowed to reach the user.

The corrected logic was:

```python
if (
    verified_trend is not None
    and verified_performance is not None
    and trend_passed
    and performance_passed
):
    print("EVALUATION PASSED")
    print(claims.model_dump())
else:
    print("EVALUATION FAILED")
    print("Answer rejected due to factual inconsistencies.")
```

This pattern is often called **fail-closed validation**: if a required check is missing or fails, the system does not present the answer as verified.

I also learned to print **`claims.model_dump()`**, the validated structured answer, instead of mistakenly printing `response.message.content`, which referred to a different model response in the agent loop.

Another bug I encountered was indentation: I accidentally placed `messages.append(...)` inside the `else` branch for one tool. That meant results from `get_plant_history()` were sometimes not returned to Qwen. I moved the append operation outside the tool-specific branches so **every executed tool sends a corresponding result message**.

**What this evaluation does *not* guarantee:** the free-text `summary` can still contain an unsupported inference even when `trend` and `performance_by_date` both pass. My evaluator verifies only the facts for which I wrote explicit checks.

---

## 4. Exercise 27: Validating arguments the LLM sends to Python

In earlier runs, Qwen requested `get_plant_history()` with limits such as `30` and `50`, even though my toy database held only three records.

There is a more general security problem here: **model-generated tool arguments are untrusted inputs**. If a model can call a function, I must still control what values that function accepts.

I added validation before execution:

```python
if function_name == "get_plant_history":
    limit = arguments.get("limit", 7)

    if type(limit) is not int:
        raise ValueError("Limit is not an integer")
    elif limit <= 0:
        raise ValueError("Limit must be positive")
    elif limit > 30:
        limit = 30

    arguments["limit"] = limit

records = function(**arguments)
```

The rules were:

| Requested limit | Application behavior |
|---|---|
| `7` | Accept 7 |
| `50` | Cap at 30 |
| `1_000_000` | Cap at 30 |
| `-5` or `0` | Reject |
| `"many"` | Reject |

Two Python details were new or useful to revisit:

```python
arguments.get("limit", 7)
```

This reads the key if present, or returns the **default value `7`** if missing. It does not use `7` when the key exists with a value of `None`.

And:

```python
arguments["limit"] = limit
```

This is the correct way to update a dictionary. I initially tried `arguments.limit = limit`, treating the dictionary like an object attribute.

I also reversed my first `isinstance()` check by mistake. `isinstance(limit, int)` is **true for integers**, not for values that aren't integers. Using `type(limit) is not int` also avoids accepting `True` or `False`, which are subclasses of `int` in Python.

For this exercise, raising `ValueError` was enough. In a production agent, I would need a controlled error response and recovery behavior instead of letting bad arguments crash the process.

**My rule now:** The LLM can *propose* actions, but application code enforces permission, validation, resource limits, and execution.

---

## 5. Exercises 28–31: My first document-based RAG pipeline

After the agent evaluations, I started learning **Retrieval-Augmented Generation (RAG)**.

My original agent could calculate *how far below target* a solar plant generated, but it did not have operational documents that explained what should be investigated. A generation shortfall alone cannot determine whether the problem was weather, equipment, curtailment, or measurement quality.

I created:

```text
knowledge/
└── solar_operations.txt
```

The sample document included this guidance:

```text
Solar Plant Operations Guide

When actual generation is below expected generation,
investigate the following possible factors:

1. Solar irradiance below forecast.
2. Inverter faults or equipment outages.
3. Grid curtailment or disconnection.
4. Incorrect meter readings or missing data.

Generation measurements alone cannot establish
which factor caused an underperformance event.

Always examine operational logs and weather data
before assigning a root cause.
```

This is **demonstration documentation**, not a record establishing what actually happened at Caucaia.

### 5.1 Reading documents with Python

I first read it with the standard library:

```python
from pathlib import Path

document = Path("knowledge/solar_operations.txt").read_text(
    encoding="utf-8"
)
```

Then I built a keyword retriever:

```python
def retrieve_information(query, document):
    matches = []

    for line in document.splitlines():
        if query.lower() in line.lower():
            matches.append(line)

    return matches
```

For `query = "inverter"`, the result was:

```python
["2. Inverter faults or equipment outages."]
```

For `query = "battery"`, it returned an empty list.

I passed the matching text into Qwen as context, along with a system instruction:

```python
from ollama import chat

context = "\n".join(retrieve_information(query, document))

response = chat(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer using only the provided context. "
                "If the context is insufficient, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ],
    think=False,
)
```

With inverter context, Qwen answered from the matching line. When the context was empty, it said there was insufficient information to answer using that context.

That was my first basic RAG pipeline:

```text
Question → Keyword retrieval → Retrieved text → Qwen → Answer
```

A detail I noticed later: when I changed the search term to `battery`, I had initially left the actual user question about inverters. That is fine as a mechanical test of empty retrieval, but in a real application the search query needs to reflect the user's information need.

### 5.2 Understanding `.strip()` and list comprehensions

I was initially confused by this pattern:

```python
lines = [
    line.strip()
    for line in document.splitlines()
    if line.strip()
]
```

The key concepts:

- `document.splitlines()` breaks the text into lines.
- `.strip()` removes leading and trailing whitespace.
- `if line.strip()` keeps only lines that still have content after stripping.
- An empty string `""` is **falsy** in a Python condition, while a nonempty string is truthy.

This is equivalent to:

```python
lines = []

for line in document.splitlines():
    if line.strip():
        lines.append(line.strip())
```

I found it helpful to learn both forms. The compact syntax is convenient once I understand the underlying loop.

---

## 6. Exercises 32–33: Embeddings and cosine similarity

Keyword matching works only when the same word or substring appears in the document. A user might instead ask:

> What could cause the electrical equipment to malfunction?

That should be related to an inverter failure, even without the word *inverter* in the question.

To search by meaning, I installed a local embedding model:

```bash
ollama pull nomic-embed-text
```

I learned to distinguish the two local models:

- **`qwen3.5:9b`**: a generative language model that produces explanations and tool requests.
- **`nomic-embed-text`**: an embedding model that converts text into numerical vectors.

### 6.1 My first embedding

```python
from ollama import embed

response = embed(
    model="nomic-embed-text",
    input="The solar inverter stopped working.",
)

vector = response.embeddings[0]

print("Dimensions:", len(vector))
print("First 10 numbers:", vector[:10])
```

My output had **768 dimensions**. Its first ten values were:

```python
[
    0.033074096, 0.015128775, -0.15546928,
    -0.007262079, 0.048863318, 0.032965176,
    0.018885838, 0.027871432, -0.010051456,
    -0.037584938,
]
```

An individual dimension is not something I should interpret as a direct label such as *fault probability*. What matters for retrieval is how the complete vector relates to other vectors generated by the same embedding model.

### 6.2 Implementing cosine similarity

I implemented the similarity function myself:

```python
import math


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.hypot(*a)
    length_b = math.hypot(*b)

    return dot_product / (length_a * length_b)
```

This computes the dot product of two vectors and divides it by the product of their magnitudes:

\[
\operatorname{cosine\_similarity}(A, B)
= \frac{A \cdot B}{\|A\|\,\|B\|}
\]

I learned that `math.hypot(*a)` can calculate the Euclidean magnitude of a vector; `*a` unpacks the list into positional arguments.

I compared sentence pairs:

| Pair | Approximate cosine similarity |
|---|---:|
| “The solar inverter stopped working.” vs. “The electrical equipment failed.” | **0.65** |
| “The solar inverter stopped working.” vs. “I love eating pizza.” | **0.25** |

The equipment-related sentences aligned much more closely than the unrelated pizza sentence.

**A cosine similarity score is not a probability**. `0.65` does not mean “65% of the words match” or “65% chance the statements are equivalent.” It's a geometric measure of vector alignment.

The exercise implementation assumes vectors have equal length and nonzero magnitude. A reusable production helper should validate those conditions.

---

## 7. Exercises 34–36: Semantic retrieval from my document

I embedded every nonempty line of `solar_operations.txt`. There were **11 lines and 11 embeddings**.

Then I embedded a question:

> What could cause the electrical equipment to malfunction?

One approach I discovered was to make a **single embedding call** with the document lines plus the question appended at the end:

```python
response = embed(
    model="nomic-embed-text",
    input=lines + [question],
)

question_vector = response.embeddings[-1]
document_vectors = response.embeddings[:-1]
```

Using `[-1]` selected the question vector; using `[:-1]` selected all preceding document vectors.

I then calculated similarity against each document line:

```python
scores = [
    cosine_similarity(question_vector, vector)
    for vector in document_vectors
]

best_score = max(scores)
best_index = scores.index(best_score)
best_line = lines[best_index]
```

My actual top result was:

```text
Best match: 2. Inverter faults or equipment outages.
Similarity: 0.7697420295142356
```

**This was the moment semantic retrieval clicked for me.** I hadn't searched for the literal word `inverter`; I had asked about malfunctioning electrical equipment, and the relevant passage ranked first.

I passed that retrieved line to Qwen. Its answer was straightforward:

> Inverter faults or equipment outages could cause the electrical equipment to malfunction.

So my pipeline became:

```text
Question
   ↓
Embedding model
   ↓
Compare question vector with document vectors
   ↓
Retrieve the best passage
   ↓
Qwen answers using the passage
```

There is an architectural improvement I want to make later: document embeddings should normally be computed **once when documents are indexed**, not recomputed for every user question.

---

## 8. Exercises 37–38: Top-K retrieval and its limits

A single passage might not contain enough evidence for a complex question. I therefore ranked all candidates and selected the most relevant `K` passages.

```python
pairs = list(zip(scores, lines))
ranked = sorted(pairs, reverse=True)
top_3 = ranked[:3]
```

`zip(scores, lines)` pairs each numerical score with the exact passage it belongs to. Sorting the pairs descending ranks the results by similarity.

For the equipment-malfunction question, the top three were:

| Rank | Score | Passage |
|---|---:|---|
| 1 | 0.7697 | Inverter faults or equipment outages |
| 2 | 0.6364 | Investigate the following possible factors |
| 3 | 0.5962 | Incorrect meter readings or missing data |

I then combined the retrieved passages:

```python
context = "\n".join(line for score, line in top_3)
```

The resulting answer included inverter faults but also mentioned meter-reading problems. That answer was consistent with the retrieved document, but it illustrated a subtle issue: **incorrect meter readings may explain an apparent production discrepancy, not a physical equipment malfunction**.

More retrieved text can offer useful context, but it also gives the model more opportunities to conflate related ideas.

Another practical lesson: I originally imported my cosine similarity helper from a previous exercise script:

```python
from exercise_32_embeddings import cosine_similarity
```

An unrelated line appeared in the terminal:

```text
Cosine similarity: 0.2581198229626706
```

That output likely came from **top-level code executing during import**. Python doesn't import only the requested function; it executes the imported module's top-level statements. A cleaner approach is to move reusable helpers into a dedicated `similarity_utils.py`, or guard demonstration code with `if __name__ == "__main__":`.

---

## 9. Exercises 39–42: Chunking changes what the model can retrieve

Line-by-line retrieval had another problem: some lines were meaningless without their neighbors.

For instance:

```text
investigate the following possible factors:
```

That line may be semantically similar to a question but does not actually list the factors.

### 9.1 Splitting by paragraphs

I tried paragraph chunking:

```python
chunks = [
    paragraph.strip()
    for paragraph in document.split("\n\n")
    if paragraph.strip()
]
```

The document became five chunks:

1. Document title.
2. Introductory sentence about below-expected generation.
3. List of four possible causes.
4. Warning that generation measurements cannot establish root cause.
5. Instruction to examine logs and weather data.

For the question:

> What could cause a solar plant to generate less energy than expected?

The initial five-chunk search produced:

| Chunk | Content | Similarity |
|---|---|---:|
| 1 | Title | 0.6408 |
| 2 | Introduction | **0.7117** |
| 3 | Actual list of possible causes | 0.6797 |
| 4 | Root-cause uncertainty | 0.6811 |
| 5 | Operational investigation guidance | 0.5068 |

The best-matching chunk contained an introduction, **not the answer**. The list of causes ranked third, so Top-2 retrieval would have omitted it.

This was an important discovery: **the most semantically similar passage is not always the most useful passage**.

### 9.2 Combining paragraphs that belong together

I manually merged the introduction and the list:

```python
combined_chunk = chunks[1] + "\n" + chunks[2]

improved_chunks = [
    chunks[0],
    combined_chunk,
    chunks[3],
    chunks[4],
]
```

I first made a mistake: my output displayed the new chunks but still showed the **old five similarity scores**. I'd updated the text list without recomputing the embeddings.

That bug was particularly dangerous because it could silently pair a score with the **wrong** chunk. I learned to verify alignment:

```python
assert len(improved_chunks) == len(similarity)
```

After re-embedding the four improved chunks, the scores became:

```python
[0.6408397720797566,
 0.7625469876509202,
 0.6810598030678585,
 0.5068038903899994]
```

The merged cause-list chunk ranked first, with similarity **0.7625**, up from **0.7117** for the old top-ranked introductory chunk.

More importantly, it now contained **the information required to answer**:

```text
When actual generation is below expected generation,
investigate the following possible factors:
1. Solar irradiance below forecast.
2. Inverter faults or equipment outages.
3. Grid curtailment or disconnection.
4. Incorrect meter readings or missing data.
```

I retrieved the top two chunks and sent both to Qwen. It correctly listed the possible causes without asserting that any specific cause had occurred at Caucaia.

I still want to improve answer quality: retrieving the root-cause caution does not force Qwen to mention that caution in its final answer.

---

## 10. Exercise 43: Writing an automatic paragraph chunker from scratch

Manual chunk-merging works on a tiny document, but it doesn't scale. I therefore implemented an algorithm to group paragraphs until the next paragraph would exceed a character limit.

I started with an incorrect version that **added a paragraph first and checked the length afterward**. The first chunk grew to **384 characters** despite a maximum of 300.

Then I moved the check earlier but accidentally duplicated paragraphs by doing both `chunk = line` and `chunk += line` in the same iteration.

The corrected algorithm checks a **candidate** before committing to it:

```python
def chunk_document(text: str, max_chars: int = 300) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = current + "\n\n" + paragraph if current else paragraph

        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
```

The central logic is:

```python
candidate = current + "\n\n" + paragraph if current else paragraph
```

`candidate` is a **hypothetical** future chunk. If it's too large, I save the current chunk and start a new one with the incoming paragraph. I do **not** save the oversized candidate.

I spent time understanding why this condition matters:

```python
if len(candidate) > max_chars and current:
```

The `and current` prevents appending an empty chunk if the first paragraph is already longer than the limit.

Another key edge case: a **single paragraph longer than `max_chars`** is allowed as one oversized chunk. That was the stated policy of this exercise; splitting an oversized paragraph into sentences or words would be a separate requirement.

### 10.1 Testing with a smaller limit

I tested with `max_chars = 100` and got:

| Chunk | Characters | Content |
|---|---:|---|
| 0 | 28 | Title |
| 1 | 96 | Introduction |
| 2 | **159** | Four possible causes |
| 3 | 93 | Root-cause warning |
| 4 | 79 | Logs and weather guidance |

The 159-character chunk is **valid under my current policy** because its paragraph already exceeds 100 characters. All other chunks stayed within the limit.

This exercise was less about AI and more about practical software engineering: testing the limits *before* mutation, avoiding duplication, handling empty inputs, and respecting an explicit contract.

It also revealed a tension: smaller chunks may be easier to rank precisely, but they can separate an explanation from its evidence. Good chunking is about preserving meaning, not only satisfying a size constraint.

---

## 11. Exercise 44: Turning semantic retrieval into a reusable function

I combined the embedding, similarity, and ranking logic into `retrieve_top_k()`.

An educational version of the final function is:

```python
from ollama import embed


def retrieve_top_k(question, chunks, k=2):
    if not chunks or k <= 0:
        return []

    response = embed(
        model="nomic-embed-text",
        input=chunks + [question],
    )

    chunk_vectors = response.embeddings[:-1]
    question_vector = response.embeddings[-1]

    similarities = [
        cosine_similarity(question_vector, vector)
        for vector in chunk_vectors
    ]

    ranked = sorted(
        zip(similarities, chunks),
        reverse=True,
    )

    top_k = ranked[:k]
    return [chunk for score, chunk in top_k]
```

Two corrections I needed during implementation:

1. I initially hardcoded `ranked[:2]` even though the function accepted an argument named `k`. That meant the parameter didn't actually work.
2. I originally returned `(score, chunk)` tuples, but my evaluation expected a list of text strings. I changed the return value to extract the passages.

Then I wrote the first retrieval test:

```python
question = "What equipment problems could explain poor solar generation?"
results = retrieve_top_k(question, chunks, k=2)

expected_evidence = "Inverter faults or equipment outages"

passed = any(
    expected_evidence in chunk
    for chunk in results
)

print("Retrieval evaluation:", "PASS" if passed else "FAIL")
```

It passed. The first retrieved passage contained the four possible causes, including inverter faults.

The new function `any()` was useful: it returns `True` if **at least one** retrieved passage contains the expected evidence.

**What this test measures:** whether the correct evidence appeared in the returned passages. It does **not** measure whether the generated answer used the evidence correctly.

---

## 12. Exercise 45: My first retrieval evaluation suite

One successful query wasn't enough, so I created a small dataset of four questions with expected evidence:

```python
test_cases = [
    {
        "question": "What equipment problems could explain poor generation?",
        "expected": "Inverter faults or equipment outages",
    },
    {
        "question": "What weather conditions could reduce solar production?",
        "expected": "Solar irradiance below forecast",
    },
    {
        "question": "Can generation measurements establish the root cause?",
        "expected": "Generation measurements alone cannot establish",
    },
    {
        "question": "What data should I examine before determining a root cause?",
        "expected": "operational logs and weather data",
    },
]
```

Then I evaluated both `k=1` and `k=2`:

```python
for k in range(1, 3):
    test_results = []

    for test_number, test in enumerate(test_cases, start=1):
        results = retrieve_top_k(test["question"], chunks, k=k)

        passed = any(
            test["expected"] in chunk
            for chunk in results
        )

        test_results.append(passed)
        print(f"Test {test_number}: {'PASS' if passed else 'FAIL'}")

    hit_rate = sum(test_results) / len(test_results)
    print(f"Hit Rate@{k}: {hit_rate:.0%}")
```

My measured results were:

| Test | Hit@1 | Hit@2 |
|---|---|---|
| Equipment problems | **FAIL** | **FAIL** |
| Weather conditions | PASS | PASS |
| Root-cause inference | PASS | PASS |
| Data to investigate | PASS | PASS |
| **Overall hit rate** | **75%** | **75%** |

Increasing K from 1 to 2 **did not improve the hit rate** for these four examples.

The metric is **Hit Rate@K**:

\[
\operatorname{HitRate@K}
= \frac{\text{questions with required evidence in Top-K}}
       {\text{total evaluation questions}}
\]

For my experiment, three of four questions passed, so Hit Rate@1 and Hit Rate@2 were both **0.75**, or **75%**.

A useful property: for a fixed ranking and the same matching rule, Hit Rate@K should not decrease when K increases. Adding more results cannot remove a previously retrieved matching passage. But it may stay unchanged, as it did here.

### 12.1 The failed test is more useful than it first looks

Earlier, a slightly different question passed:

> What equipment problems could explain **poor solar generation**?

But this test failed:

> What equipment problems could explain **poor generation**?

The difference in wording is small, yet retrieval behavior changed. I haven't diagnosed the exact cause yet, but it demonstrates why evaluating one carefully chosen prompt is not enough.

Possible areas to investigate include query wording, chunk boundaries, embeddings, and the ranking of the cause-list chunk. I should inspect actual retrieved passages **before** deciding which one is responsible.

### 12.2 Limits of the current evaluation

My evaluation suite is a starting point, not a robust benchmark:

- **Only four questions**, all based on one tiny sample document.
- A **case-sensitive substring match** decides PASS or FAIL; legitimate paraphrases or formatting changes could cause false failures.
- One piece of expected evidence is checked per question, even when a complete answer may require several pieces.
- It evaluates **retrieval**, not answer correctness, citation quality, or unsupported claims.
- I recompute document embeddings on each query, which is inefficient and unsuitable for a larger application.
- The 100-character chunker preserves oversized paragraphs but doesn't address paragraph overlap, sentence-aware splitting, or metadata such as source filenames and page numbers.

Nevertheless, it gives me a **repeatable measurement**. I can change a retrieval strategy and compare the results instead of relying on whether one model answer sounds convincing.

---

## 13. Lessons about engineering, not just AI

Several of my most useful mistakes were ordinary software engineering mistakes:

| Mistake or surprise | What it taught me |
|---|---|
| Pydantic model included only `trend` | Validate the schema itself, not only model output. |
| `verified_trend` remained `None` | Calculation and state persistence are different steps. |
| Printed claims before checking `passed` | Evaluating a result is not the same as enforcing the result. |
| Tool-result message was indented under one branch | Every tool call needs its own result message. |
| Used `arguments.limit` on a dictionary | Use dictionary key syntax: `arguments["limit"]`. |
| Reversed `isinstance()` condition | Read boolean predicates carefully; write rejection tests. |
| Added paragraph before checking chunk length | Validate a candidate before committing state changes. |
| Duplicated a paragraph while starting a new chunk | Ensure each input item is consumed exactly once. |
| Changed chunks without recomputing embeddings | Keep every embedding aligned with its original source text. |
| Imported a practice module and saw unexpected output | Imports execute top-level module code. |
| Hardcoded Top-2 despite accepting `k` | Test whether configurable parameters actually affect behavior. |
| Returned score/text tuples to a test expecting strings | Define clear function interfaces and return contracts. |

I'm trying to improve my Python fluency while learning applied AI. This seems especially important if I want to work at a startup, where I may need to build, debug, deploy, and support a product with relatively little hand-holding.

---

## 14. What I can build now—and what I still cannot claim

With these exercises, I now understand and have implemented the core pieces of two connected systems.

### A. A tool-calling energy analyst with deterministic checks

```text
User question
    ↓
Qwen selects approved Python tools
    ↓
Python validates arguments and queries SQLite
    ↓
Python computes verified trend/performance
    ↓
Qwen returns structured claims
    ↓
Pydantic validates structure
    ↓
Python compares claims to ground truth
    ↓
Accept verified classifications or reject the response
```

### B. A semantic RAG pipeline with basic retrieval evaluation

```text
Operational text document
    ↓
Paragraph-aware chunking
    ↓
Embedding model (nomic-embed-text)
    ↓
Question embedding + cosine similarity
    ↓
Top-K ranked passages
    ↓
Qwen generates an answer from retrieved context

Separate test loop:
Question + expected passage → Retrieve Top-K → PASS/FAIL → Hit Rate@K
```

What I **cannot** claim yet:

- That either system is deployed or production-ready.
- That structured-output validation proves the entire generated summary is correct.
- That a 75% hit rate on four toy examples represents real-world reliability.
- That the retriever can efficiently search thousands of documents.
- That the agent can safely recover from all tool exceptions, malformed results, or malicious document instructions.

The aim is to close those gaps incrementally rather than hide them with a convincing demo.

---

## 15. Next steps

I stopped immediately before **Exercise 46**, which is about diagnosing the retrieval test that failed at both `k=1` and `k=2`.

The next debugging action is to inspect the **actual Top-3 retrieved chunks** for the failed question:

```python
question = test_cases[0]["question"]
results = retrieve_top_k(question, chunks, k=3)

for rank, chunk in enumerate(results, start=1):
    print(f"\nRank {rank}:")
    print(chunk)
```

I don't want to tune embeddings, increase K, change the question, or modify chunking before I see the failure. **A metric tells me something went wrong; it doesn't tell me why.**

After that, I want to move toward a more realistic RAG implementation: reusable module structure, cached/indexed embeddings, a vector database, source metadata, retrieval-quality tests with more varied questions, answer-level grounding checks, and eventually deployment and monitoring.

My longer-term goal is to turn the Solar Plant AI Analyst into a credible portfolio project for **Applied AI Engineer or AI-focused software engineering roles at US startups**. To get there, I need more than functioning examples. I need software that is understandable, testable, observable, reasonably safe, and useful to someone other than me.

## Final takeaway

My understanding of the engineering problem has changed across these exercises:

> **An LLM's fluent answer is not proof of correctness. A high similarity score is not proof of useful retrieval. And a passing test is only as meaningful as the behavior that test actually measures.**

The most valuable habit I'm developing is to separate the parts I can compute or verify deterministically from the parts where a language model adds value—and to test the boundaries between them.

That's the direction I want to keep following.
