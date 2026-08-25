# llm_engine.py

Standalone module for async batch LLM tasks via OpenRouter. Copy to any project, install deps, and go.

## API

### `setup_env()`

Loads the environment file; asserts `OPENROUTER_API_KEY` exists.

### `openrouter_chat(client, model, messages, response_model=None, num_retries=3, **extra_body)`

One OpenRouter chat completion. `model` is an OpenRouter id (`"google/gemini-3.6-flash"`). `response_model` becomes a strict `json_schema` response format. Extra keywords (`tools=`, `temperature=`, …) go into the request body. Retries 408/429/5xx, network errors, and `{"error": ...}` payloads with backoff.

Returns `ModelResponse`: `choices[0].message.content`, `usage.cost`, `response_ms`. Provider extras (e.g. web-search `annotations`) are on `message.model_extra`.

### `build_response_format(response_model)`

The strict `json_schema` `response_format` dict; used by `openrouter_chat` and by batch request bodies.

### `make_client()`

`httpx.AsyncClient` with no connection cap. `run_batch` opens one per batch; for one-off calls: `async with make_client() as client:`.

### `load_and_prepare(data_dir, input_file, outfile, models, id_cols, response_model, out_dir=None)`

Reads `.jsonl`/`.csv`, cross-joins with `models`, anti-joins against `outfile` to drop processed rows (filters derived from `response_model`: `Literal` -> `is_in`, else `is_not_null`). Returns `list[dict]`. `out_dir` defaults to `data_dir`.

### `run_batch(dat, get_response, response_model, out_dir=None, outfile=None, batch_size=200, request_timeout=180.0, ...)`

Calls `get_response(semaphore, data, client)` per row under a semaphore of `batch_size`, cancelling after `request_timeout`s. Validates with `model_validate_json()`, appends successes to `outfile` and failures to `*_errors.jsonl` (with `outfile=None`, prints every row as JSON instead), shows cost and progress via tqdm. `debug_batch_size > 0` processes only that many rows and logs each response.

### `run_batch_api(dat, build_request, response_model, out_dir, outfile, id_cols, debug_batch_size=0)`

Same task as `run_batch()` through OpenRouter's Batch API: 50% token price, results within 24h, text only, no per-row retries. Each call collects finished batches, then submits one batch per model for rows not in flight. Details under [Batch API](#batch-api). Lower level: `submit_batch(model, requests)`, `get_batch(batch_id)`.

## Basic example: classify with multiple models

```python
import asyncio
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from src.llm_engine import load_and_prepare, openrouter_chat, run_batch, setup_env

setup_env()

DATA_DIR = Path("data/my-dataset/processed")
OUTFILE = "classified.jsonl"
MODELS = ["openai/gpt-5.4", "anthropic/claude-opus-4.6"]
ID_COLS = ["id", "model"]
N_RETRIES = 5


class Response(BaseModel):
    label: Literal["true", "false", "unsure"] = Field(...)
    explanation: str = Field(...)


dat = load_and_prepare(DATA_DIR, "input.jsonl", OUTFILE, MODELS, ID_COLS, Response)


async def get_llm_response(semaphore, data, client):
    async with semaphore:
        try:
            response = await openrouter_chat(
                client,
                model=data["model"],
                messages=[{"role": "user", "content": data["content"]}],
                response_model=Response,
                num_retries=N_RETRIES,
            )
            return response, data, response.usage.cost
        except Exception as e:
            return e, data, 0


asyncio.run(
    run_batch(dat, get_llm_response, Response, DATA_DIR, OUTFILE, batch_size=200)
)
```

## Skipping `load_and_prepare()`: custom data loading

Build `dat` yourself and pass it to `run_batch()` (see `src/process/factcheck_inline.py` for a full example that also prints instead of writing):

```python
setup_env()

# Custom data loading
dat = []
with open("input.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row.get("content"):
            dat.append({"id": row["id"], "content": row["content"]})

# Custom dedup
already_done = [f.stem for f in OUT_DIR.glob("*.json")]
dat = [x for x in dat if x["id"] not in already_done]

asyncio.run(run_batch(dat, get_llm_response, Response, OUT_DIR, OUTFILE, batch_size=20))
```

## Two-step LLM calls (Gemini search)

`get_llm_response` may make several calls; sum the cost:

```python
async def get_llm_response(semaphore, data, client):
    async with semaphore:
        cost = 0.0
        try:
            # Step 1: search with Gemini + Google Search grounding
            search_response = await openrouter_chat(
                client,
                model=data["model"],
                messages=[{"role": "user", "content": data["content"]}],
                num_retries=5,
                web_search_options={"search_context_size": "medium"},
            )
            cost += search_response.usage.cost
            search_text = search_response.choices[0].message.content

            # Step 2: extract structured JSON from the search result
            response = await openrouter_chat(
                client,
                model="google/gemini-3.1-flash-lite-preview",
                messages=[
                    {"role": "user", "content": f"Extract as JSON:\n\n{search_text}"}
                ],
                response_model=Response,
            )
            cost += response.usage.cost
            return response, data, cost
        except Exception as e:
            return e, data, 0
```

## Custom output with `on_success`

Replace the default JSONL append with a callback; validation, error handling, and cost tracking are unchanged:

```python
async def save_annotations(data: dict, response: ModelResponse, out_dir: Path) -> None:
    """Extract web search annotations and save as per-item JSON."""
    annotations = response.choices[0].message.model_extra["annotations"]
    async with aiofiles.open(out_dir / f"{data['id']}.json", mode="a+") as f:
        await f.write(json.dumps(annotations) + "\n")


asyncio.run(
    run_batch(
        dat,
        get_llm_response,
        Response,
        OUT_DIR,
        OUTFILE,
        on_success=save_annotations,
        batch_size=300,
    )
)
```

## Debug mode

One knob, `debug_batch_size` (default `0` = off). In `run_batch()`, `> 0` truncates `dat` to that many rows and logs each response's status, cost, and parsed fields. In `run_batch_api()`, `> 0` caps rows in flight (pending + new) at that number, so re-running while a debug batch is pending submits nothing.

```python
asyncio.run(run_batch(dat, get_llm_response, Response, DATA_DIR, OUTFILE, debug_batch_size=5))
```

## Batch API

`run_batch_api()` is the Batch-API counterpart of `run_batch()`. Same inputs (`dat` from `load_and_prepare`, a `Response` model, `OUT_DIR`/`OUTFILE`), same output files, so both paths can be mixed on one dataset. Instead of `get_llm_response`, supply `build_request(row) -> body` returning the request body (`messages`, `tools`, …); the engine adds `model` and `response_format`.

```python
def build_request(data: dict) -> dict:
    return {"messages": [{"role": "user", "content": data["content"]}]}


run_batch_api(dat, build_request, Response, OUT_DIR, OUTFILE, ID_COLS)
```

Each run does two things, in order:

1. **Collect.** For every batch in `<OUTFILE>.batches.json` whose rows are not all on disk, `GET` it. Non-terminal (`validating`/`in_progress`/`finalizing`): log status and counts. `completed`: validate each result with `Response`, append successes to `OUTFILE` and failures to `*_errors.jsonl` (rows carry `batch_id`, `pulltime`, `cost`), writing only rows not already present. `failed`/`expired`/`cancelled`: write every row to the error file so the batch is skipped next time and the rows retry through the normal dedup path.
2. **Submit.** Rows in `dat` that are not in a pending batch and were not just collected are grouped by model and submitted, one batch per model. The registry records `batch_id`, `model`, `submitted_at`, and `custom_id -> input row` (`custom_id` = JSON of the row's `ID_COLS` values, so `ID_COLS` must uniquely identify a row).

Run the script once to submit, again later (within 24h) to collect; re-running is always safe. The registry is append-only and "done" is derived from the output files, so deleting `OUTFILE` (or a single row in it) rebuilds it from the stored batches — results are kept by OpenRouter for 30 days — without resubmitting or paying again. A batch cannot be modified after submission; more rows mean another batch.

Constraints: text only; one model per batch (handled); on Google models every request in a batch must share one `response_format` (true here since one `Response` per script). Web search is billed at the normal rate.

Full example: `src/process/factcheck_batch.py` — identical to `factcheck.py` except `build_request` replaces `get_llm_response`.
