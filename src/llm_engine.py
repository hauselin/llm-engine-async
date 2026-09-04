"""
LLM processing engine. Refer to llm_engine.md for details and examples.

Shared infrastructure for async LLM tasks (batch or single):
- setup_env(): Load .env and verify the OpenRouter API key
- openrouter_chat(): Async chat completion against OpenRouter's REST API with retries
- run_batch_api(): Batch-API counterpart of run_batch() (50% price, 24h window)
- load_and_prepare(): Load input data, cross-join with models, deduplicate
- run_batch(): Async processor with concurrency control, progress tracking, JSONL output

Dependencies:
    uv add polars httpx aiofiles loguru pydantic python-dotenv tqdm
"""

# %%

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import aiofiles
import httpx
import polars as pl
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_BATCH_URL = "https://openrouter.ai/api/beta/batches"
BATCH_TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


# %% OpenRouter response types (OpenAI-compatible; unknown fields are kept)


class Message(BaseModel, extra="allow"):
    role: str
    content: str | None = None


class Choice(BaseModel, extra="allow"):
    message: Message
    finish_reason: str | None = None


class Usage(BaseModel, extra="allow"):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0  # populated when the request sends usage={"include": true}


class ModelResponse(BaseModel, extra="allow"):
    id: str
    model: str
    choices: list[Choice]
    usage: Usage = Usage()
    response_ms: float = -1.0  # set client-side, not part of the API payload


class OpenRouterError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status


# Type alias for the async get_response callable that each script provides
GetResponseFn = Callable[
    [asyncio.Semaphore, dict, httpx.AsyncClient],
    Coroutine[Any, Any, tuple[ModelResponse | Exception, dict, float]],
]

# Type alias for optional on_success callback
OnSuccessFn = Callable[
    [dict, ModelResponse, Path],
    Awaitable[None],
]


def setup_env() -> None:
    """Load .env and verify the OpenRouter API key is present."""
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not found in environment")


def make_client() -> httpx.AsyncClient:
    """Create the shared HTTP client; use as `async with make_client() as client:`.

    No pool cap: httpx's default 100 connections + 5s pool timeout would raise
    PoolTimeout under batch_size > 100; run_batch's semaphore is the only limit.
    The read timeout is deliberately long: run_batch's request_timeout is the
    per-row limit, so httpx only guards against a socket that never answers.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=10.0),
        limits=httpx.Limits(max_connections=None),
    )


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def build_response_format(response_model: type[BaseModel]) -> dict[str, Any]:
    """Strict json_schema response_format shared by the sync and batch paths."""
    # OpenAI-family strict mode rejects schemas without additionalProperties=false
    schema = response_model.model_json_schema() | {"additionalProperties": False}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "strict": True,
            "schema": schema,
        },
    }


# %% Batch API (https://openrouter.ai/docs/batch-quickstart)


def submit_batch(
    model: str, requests: list[dict], endpoint: str = "/v1/chat/completions"
) -> dict:
    """Submit one batch; returns the batch object (status "validating").

    requests: [{"custom_id": <unique str>, "body": {"messages": [...], ...}}, ...]
    Key order matters: the server stream-parses and rejects bodies where
    "requests" precedes "endpoint"/"model".
    """
    assert requests, "requests is empty"
    ids = [r["custom_id"] for r in requests]
    assert len(set(ids)) == len(ids), "custom_id values must be unique within a batch"
    body = {
        "endpoint": endpoint,
        "model": model.removeprefix("openrouter/"),
        "requests": requests,
    }
    r = httpx.post(
        OPENROUTER_BATCH_URL, json=body, headers=_auth_headers(), timeout=300.0
    )
    if r.status_code not in (200, 202):
        raise OpenRouterError(r.status_code, r.text)
    return r.json()


def get_batch(batch_id: str) -> dict:
    """Fetch a batch; when status is "completed", "results" is inline."""
    r = httpx.get(
        f"{OPENROUTER_BATCH_URL}/{batch_id}", headers=_auth_headers(), timeout=300.0
    )
    if r.status_code != 200:
        raise OpenRouterError(r.status_code, r.text)
    return r.json()


# Script-provided: row dict -> request body (messages, tools, ...); response_format is added
BuildRequestFn = Callable[[dict], dict]


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_batch_api(
    dat: list[dict],
    build_request: BuildRequestFn,
    response_model: type[BaseModel],
    out_dir: Path,
    outfile: str,
    id_cols: list[str],
    debug_batch_size: int = 0,
) -> None:
    """Batch-API counterpart of run_batch(); call it repeatedly until nothing is pending.

    Each call: (1) fetches every batch recorded in <outfile>.batches.json and, for
    those completed, validates results and appends rows to outfile / *_errors.jsonl
    exactly like run_batch(); (2) submits one batch per model for rows in dat that
    are not already in flight. Rows in the registry are keyed by id_cols. A batch
    is considered done when all its rows are on disk, so deleting outfile
    regenerates it from the batches (results are kept 30 days).

    Args:
        dat: Rows to process (typically from load_and_prepare, so already deduped).
        build_request: row dict -> request body; response_format is added from response_model.
        response_model: Pydantic model for validating results.
        out_dir, outfile: Same as run_batch(); results land in the same file.
        id_cols: Columns that identify a row (used as the batch custom_id).
        debug_batch_size: If > 0, keep at most this many rows in flight (pending + new).
    """
    assert out_dir.is_dir(), f"Output directory does not exist: {out_dir}"
    registry_path = out_dir / f"{outfile}.batches.json"
    error_path = out_dir / f"{Path(outfile).stem}_errors.jsonl"
    registry: list[dict] = (
        json.loads(registry_path.read_text()) if registry_path.exists() else []
    )

    def key(row: dict) -> str:
        # some providers restrict custom_id to [A-Za-z0-9_-]; hash the id values
        # (the registry keeps the full row under each id, so nothing is lost)
        raw = json.dumps([row[col] for col in id_cols])
        return hashlib.sha1(raw.encode()).hexdigest()

    def written_keys(path: Path) -> set[str]:
        if not path.exists():
            return set()
        return {key(json.loads(line)) for line in path.read_text().splitlines() if line}

    # --- 1. collect: write rows from completed batches that are not yet on disk ---
    # (so deleting outfile regenerates it from the batches, without duplicates)
    on_disk = written_keys(out_dir / outfile) | written_keys(error_path)
    done_keys: set[str] = set()  # succeeded this run
    in_flight: set[str] = set()  # in a batch that is not terminal yet
    for entry in registry:
        if set(entry["rows"]) <= on_disk:
            continue
        batch = get_batch(entry["batch_id"])
        status = batch["status"]
        if status not in BATCH_TERMINAL_STATUSES:
            logger.info(
                f"{entry['batch_id']} ({entry['model']}): {status} {batch['request_counts']}"
            )
            in_flight |= set(entry["rows"])
            continue
        pulltime = datetime.now().astimezone().isoformat(timespec="seconds")
        if status != "completed":
            # write every row as an error so the batch is skipped next run and
            # the rows are retried through the normal dedup path
            logger.error(f"{entry['batch_id']}: {status} — {batch.get('error')}")
            for cid, data in entry["rows"].items():
                if cid in on_disk:
                    continue
                data |= {
                    "pulltime": pulltime,
                    "batch_id": entry["batch_id"],
                    "cost": 0.0,
                }
                data["request_status"] = f"EXCEPTION: batch {status}"
                _append_jsonl(error_path, data)
                on_disk.add(cid)
            continue
        n_success = n_written = 0
        results = batch["results"] or []
        # per-request bodies carry tokens but no cost; apportion batch cost by token share
        batch_cost = (batch.get("usage") or {}).get("cost", 0.0)
        batch_tokens = sum(
            r.get("response", {})
            .get("body", {})
            .get("usage", {})
            .get("total_tokens", 0)
            for r in results
        )
        for result in results:
            cid = result["custom_id"]
            data = entry["rows"][cid]
            data["pulltime"] = pulltime
            data["batch_id"] = entry["batch_id"]
            data["cost"] = 0.0
            if result.get("error"):
                data["request_status"] = f"EXCEPTION: {json.dumps(result['error'])}"
                target = error_path
            else:
                body = result["response"]["body"]
                tokens = body.get("usage", {}).get("total_tokens", 0)
                data["cost"] = (
                    batch_cost * tokens / batch_tokens if batch_tokens else 0.0
                )
                try:
                    data |= response_model.model_validate_json(
                        body["choices"][0]["message"]["content"] or ""
                    ).model_dump(mode="json")
                    data["request_status"] = "success"
                    target = out_dir / outfile
                except ValidationError as e:
                    data["request_status"] = f"EXCEPTION: Validation error: {e}"
                    target = error_path
            if data["request_status"] == "success":
                done_keys.add(cid)
                n_success += 1
            if cid not in on_disk:
                _append_jsonl(target, data)
                on_disk.add(cid)
                n_written += 1
        logger.success(
            f"{entry['batch_id']} ({entry['model']}): {n_success}/{len(results)} success,"
            f" {n_written} rows written, cost ${batch_cost:.4f}"
        )
    registry_path.write_text(json.dumps(registry, indent=1))

    # --- 2. submit rows not in flight and not just collected ---
    todo = [row for row in dat if key(row) not in in_flight | done_keys]
    if debug_batch_size > 0:  # cap rows in flight (pending + new)
        todo = todo[: max(0, debug_batch_size - len(in_flight))]
        logger.debug(f"DEBUG MODE: {len(todo)} new rows, {len(in_flight)} in flight")
    if not todo:
        logger.info(f"Nothing to submit ({len(in_flight):,} rows still in flight)")
        return
    response_format = build_response_format(response_model)
    for model in sorted({row["model"] for row in todo}):
        rows = [row for row in todo if row["model"] == model]
        requests = [
            {
                "custom_id": key(row),
                "body": build_request(row) | {"response_format": response_format},
            }
            for row in rows
        ]
        batch = submit_batch(model, requests)
        registry.append(
            {
                "batch_id": batch["id"],
                "model": model,
                "submitted_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "rows": {key(row): row for row in rows},
            }
        )
        registry_path.write_text(json.dumps(registry, indent=1))  # after each submit
        logger.success(f"{model}: submitted {len(rows):,} rows as batch {batch['id']}")
    logger.info(f"Batch registry: {registry_path}")


async def openrouter_chat(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    response_model: type[BaseModel] | None = None,
    num_retries: int = 3,
    **extra_body: Any,
) -> ModelResponse:
    """Call OpenRouter chat completions directly.

    Args:
        client: Open httpx.AsyncClient (run_batch supplies one; see make_client()).
        model: OpenRouter model id, e.g. "google/gemini-3.6-flash".
        messages: OpenAI-style message dicts.
        response_model: Pydantic model sent as a strict json_schema response_format.
        num_retries: Retries on 408/429/5xx and network errors (exponential backoff).
        **extra_body: Extra request-body fields (e.g. tools, temperature).
    """
    body: dict[str, Any] = {
        "model": model.removeprefix("openrouter/"),
        "messages": messages,
        "usage": {"include": True},  # ask OpenRouter to return cost
        **extra_body,
    }
    if response_model is not None:
        body["response_format"] = build_response_format(response_model)
    headers = _auth_headers()

    last_error: Exception | None = None
    for attempt in range(num_retries + 1):
        if attempt > 0:
            await asyncio.sleep(min(2**attempt, 30))
        start = time.perf_counter()
        try:
            r = await client.post(OPENROUTER_URL, json=body, headers=headers)
        except httpx.HTTPError as e:
            last_error = e
            logger.warning(f"Network error (attempt {attempt + 1}): {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000
        if r.status_code != 200:
            err = OpenRouterError(r.status_code, r.text)
        else:
            try:
                payload = r.json()
            except ValueError:  # e.g. an HTML error page from a proxy
                err = OpenRouterError(0, f"non-JSON body: {r.text}")
            else:
                if "error" not in payload:
                    response = ModelResponse.model_validate(payload)
                    response.response_ms = elapsed_ms
                    return response
                # OpenRouter proxies upstream failures as 200 + error object;
                # the real status is in error.code (missing -> treat as retryable)
                err = OpenRouterError(
                    payload["error"].get("code") or 0, json.dumps(payload["error"])
                )
        if not _is_retryable(err.status):
            raise err
        last_error = err
        logger.warning(f"Retryable error (attempt {attempt + 1}): {err}")
    assert last_error is not None
    raise last_error


def _is_retryable(status: int) -> bool:
    return status in (0, 408, 429) or status >= 500


def _build_validation_filters(response_model: type[BaseModel]) -> list[pl.Expr]:
    """Derive polars filter expressions from a Pydantic model's field types.

    - Literal fields -> is_in(allowed_values)
    - All other fields -> is_not_null()

    These filters are intentionally loose (not range-checking ints, etc.)
    because the Pydantic model validates before writing. The filters here
    just need to identify rows that were successfully processed.
    """
    filters: list[pl.Expr] = []
    for field_name, field_info in response_model.model_fields.items():
        annotation = field_info.annotation
        if get_origin(annotation) is Literal:
            allowed = list(get_args(annotation))
            filters.append(pl.col(field_name).is_in(allowed))
        else:
            filters.append(pl.col(field_name).is_not_null())
    return filters


def load_and_prepare(
    data_dir: Path,
    input_file: str,
    outfile: str,
    models: list[str],
    id_cols: list[str],
    response_model: type[BaseModel],
    shuffle: bool = True,
    extra_filters: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    out_dir: Path | None = None,
) -> list[dict]:
    """Load input data, cross-join with models, remove already-processed rows.

    Supports .jsonl and .csv input files (detected by extension).

    Args:
        data_dir: Directory containing input files.
        input_file: Input filename (.jsonl or .csv).
        outfile: Output JSONL filename (checked for already-processed rows).
        models: List of model identifiers to cross-join.
        id_cols: Columns used to match processed rows (anti-join keys).
        response_model: Pydantic model whose fields derive validation filters.
        shuffle: Shuffle rows before returning.
        extra_filters: Optional function to apply additional filters to
            already-processed data (e.g., tighter range checks).
        out_dir: Directory for output files. Defaults to data_dir if not provided.

    Returns:
        List of row dicts ready for processing. Returns empty list if nothing to process.
    """
    if out_dir is None:
        out_dir = data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / input_file
    assert data_file.exists(), f"Input file not found: {data_file}"
    if data_file.suffix == ".csv":
        d1 = pl.read_csv(data_file, infer_schema_length=100_000)
    else:
        d1 = pl.read_ndjson(data_file)
    assert d1.height > 0, f"Input file is empty: {data_file}"

    assert len(models) > 0, "Models list is empty"
    d1 = d1.join(pl.DataFrame({"model": models}), how="cross")

    # id_cols must exist in the data after cross-join
    missing_cols = [col for col in id_cols if col not in d1.columns]
    assert not missing_cols, f"ID columns not found in data: {missing_cols}"

    if shuffle:
        d1 = d1.sample(fraction=1.0, shuffle=True)

    # Remove already processed rows
    out_path = out_dir / outfile
    logger.info(f"Rows in original dataset: {d1.height:,}")

    if out_path.exists():
        try:
            d0_processed = pl.read_ndjson(out_path).filter(
                pl.col("request_status") == "success"
            )
            for f in _build_validation_filters(response_model):
                d0_processed = d0_processed.filter(f)
            if extra_filters is not None:
                d0_processed = extra_filters(d0_processed)

            if d0_processed.height > 0:
                output_csv = out_path.with_suffix(".csv")
                d0_processed.write_csv(output_csv)
                logger.info(
                    f"Save processed data to csv: {output_csv} (for downstream analysis)"
                )
                d1 = d1.join(d0_processed, on=id_cols, how="anti")
                logger.info(f"Remove {d0_processed.height:,} rows already processed")
        except Exception as e:
            logger.error(f"Error reading {out_path}: {e}")
            raise RuntimeError(f"Failed to read processed output: {out_path}") from e
        logger.info(f"Rows in dataset yet to process: {d1.height:,}")
    else:
        logger.info("No data processed yet. Starting from scratch...")

    if d1.height == 0:
        logger.warning("No tasks to process.")
        return []

    return d1.rows(named=True)


async def run_batch(
    dat: list[dict],
    get_response: GetResponseFn,
    response_model: type[BaseModel],
    out_dir: Path | None = None,
    outfile: str | None = None,
    error_file: str | None = None,
    batch_size: int = 200,
    debug_batch_size: int = 0,
    on_success: OnSuccessFn | None = None,
    request_timeout: float = 180.0,
) -> None:
    """Async batch processor with concurrency control and progress tracking.

    Manages the task queue, validates responses against response_model,
    and writes results to JSONL. The LLM call logic (prompt, model,
    search strategy) is entirely encapsulated in get_response.

    Args:
        dat: List of row dicts to process.
        get_response: Async callable (semaphore, data, client) -> (response, data, cost).
        response_model: Pydantic model for validating LLM responses.
        out_dir: Directory for output files. Required when outfile or on_success is set.
        outfile: Output JSONL filename. If None, every row (success or error) is
            printed as JSON instead of written.
        error_file: Error JSONL filename. Defaults to outfile with _errors suffix.
        batch_size: Max concurrent requests (semaphore size).
        debug_batch_size: If > 0, process only this many rows and log each response.
        on_success: Optional async callback(data, response, out_dir) on successful
            validation. Defaults to appending the row dict to out_dir/outfile.
        request_timeout: Per-request timeout in seconds. On timeout the task is
            cancelled, the semaphore slot is released, and the row is routed to
            the error file.
    """
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    assert request_timeout > 0, (
        f"request_timeout must be positive, got {request_timeout}"
    )
    if outfile is not None or on_success is not None:
        assert out_dir is not None and out_dir.is_dir(), (
            f"Output directory does not exist: {out_dir}"
        )
    if outfile is not None:
        if error_file is None:
            error_file = f"{Path(outfile).stem}_errors.jsonl"
        assert error_file != outfile, "error_file must differ from outfile"

    async def _append(filename: str | None, row: dict) -> None:
        # outfile=None means "print instead of persist"
        if filename is None or out_dir is None:
            print(json.dumps(row))
            return
        async with aiofiles.open(out_dir / filename, mode="a+", encoding="utf-8") as f:
            await f.write(json.dumps(row) + "\n")

    debug = debug_batch_size > 0
    if debug:
        dat = dat[:debug_batch_size]
        logger.debug(f"DEBUG MODE: {debug_batch_size} tasks")

    n_tasks = len(dat)
    if n_tasks == 0:
        logger.warning("No tasks to process.")
        return

    n_success = 0
    n_processed = 0
    total_cost = 0.0
    projected_total_cost = 0.0
    semaphore = asyncio.Semaphore(batch_size)
    pending: set[asyncio.Task] = set()

    logger.info(
        f"Starting session with {n_tasks:,} total requests"
        f" (concurrency: {batch_size}, timeout: {request_timeout}s)"
    )
    logger.info(f"OUTPUT: {outfile or 'stdout'}")
    logger.info("Starting in 3 seconds...")
    await asyncio.sleep(3)

    async with make_client() as client:

        async def _guarded(row: dict) -> tuple[ModelResponse | Exception, dict, float]:
            try:
                return await asyncio.wait_for(
                    get_response(semaphore, row, client), timeout=request_timeout
                )
            except TimeoutError:
                return (TimeoutError(f"request exceeded {request_timeout}s"), row, 0.0)

        with tqdm(total=n_tasks, unit="req", dynamic_ncols=True) as pbar:
            for current_data in dat[:batch_size]:
                pending.add(asyncio.create_task(_guarded(current_data)))
            next_idx = min(batch_size, n_tasks)

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for finished_job in done:
                    response, data, cost = await finished_job
                    pbar.update(1)
                    n_processed = pbar.n

                    if next_idx < n_tasks:
                        # add one more task to the queue now that one is finished
                        pending.add(asyncio.create_task(_guarded(dat[next_idx])))
                        next_idx += 1

                    # process response
                    success = False
                    data["pulltime"] = (
                        datetime.now().astimezone().isoformat(timespec="seconds")
                    )
                    data["cost"] = cost
                    total_cost += cost
                    projected_total_cost = total_cost / n_processed * n_tasks

                    if not isinstance(response, ModelResponse):
                        data["request_status"] = f"EXCEPTION: {response}"
                    else:
                        data["response_ms"] = response.response_ms
                        try:
                            ai_response = response_model.model_validate_json(
                                response.choices[0].message.content or ""
                            ).model_dump(mode="json")
                            data |= ai_response
                            data["request_status"] = "success"
                        except ValidationError as e:
                            logger.warning(f"Validation error: {e}")
                            data["request_status"] = f"EXCEPTION: Validation error: {e}"
                        else:
                            try:
                                if on_success is not None:
                                    assert out_dir is not None
                                    await on_success(data, response, out_dir)
                                else:
                                    await _append(outfile, data)
                                n_success += 1
                                success = True
                            except (OSError, ValueError, TypeError, KeyError) as e:
                                # disk/write failures and the usual on_success bugs;
                                # anything else propagates and stops the batch
                                logger.error(
                                    f"Failed to persist successful response: {e}"
                                )
                                data["request_status"] = (
                                    f"EXCEPTION: persistence error: {e}"
                                )

                    if not success:
                        await _append(error_file, data)

                    if debug:
                        shown = ["request_status", "cost", *response_model.model_fields]
                        logger.debug(
                            f"AI response: { {k: data.get(k) for k in shown} }"
                        )

                    pbar.set_postfix_str(
                        f"ok {n_success / n_processed:.0%}"
                        f" | ${total_cost:.4f} → ${projected_total_cost:.4f}"
                    )

    logger.success(
        f"Finished {n_processed:,} ({n_success:,} success) of {n_tasks:,}."
        f" Total cost: ${total_cost:.10f}"
        f" (projected: ${projected_total_cost:.4f})"
    )
