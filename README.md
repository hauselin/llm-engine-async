# llm-engine-async

Example code for calling OpenRouter models asynchronously and in batch mode. See examples in `src/process`.

## 1. Install

A Python project managed with [uv](https://docs.astral.sh/uv/), a package
manager that replaces `pip` and `venv`. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root:

```bash
uv sync
```

That reads `pyproject.toml`, downloads Python 3.14 if you lack it, creates a
`.venv/` folder, and installs every dependency at the exact version pinned in
`uv.lock`. Nothing is installed system-wide, and you never activate the
environment yourself — `uv run` does it for you.

## 2. API key

Create a `.env` file in the repo root:

```
OPENROUTER_API_KEY=sk-or-...
```

Get a key at https://openrouter.ai/keys.

## 3. Run

```bash
uv run python src/process/factcheck.py
```

Reads `data/processed/example-data.csv`, writes one JSON line per row to
`data/processed/example-data-factcheck.jsonl` (plus a `.csv` copy), and any
failures to `..._errors.jsonl`.

`example-data.csv` is a throwaway demo file: 10 rows of made-up chatbot replies,
written to show the expected input format and to confirm the pipeline works
end to end. Every row is `role = assistant` — user and system turns are already
filtered out, since only the assistant's claims get fact-checked. Replace it
with your own data.

Safe to re-run: already-processed rows are skipped, so an interrupted run
resumes where it stopped.

### Batch mode (half price, results within 24h)

```bash
uv run python src/process/factcheck_batch.py   # submits rows not yet processed
# ... later, within 24h ...
uv run python src/process/factcheck_batch.py   # collects finished results
```

Same input, same settings, same output files as `factcheck.py`, but rows go
through OpenRouter's Batch API at 50% of the token price. Each run first
collects any finished batches into the output file, then submits whatever is
still unprocessed; batch ids are tracked in
`data/processed/example-data-factcheck.jsonl.batches.json`. Re-run as often as
you like — nothing is submitted twice and rows are never written twice. Use
this when you have many rows and no need for immediate results; use
`factcheck.py` for small runs or quick iteration.

## Using your own data

Edit the settings block near the top of `src/process/factcheck.py`:

| Setting | What it does |
|---|---|
| `INPUT_FILE` | your `.csv` inside `DATA_DIR` |
| `CONTENT_COL` | column holding the text to check |
| `ID_COLS` | columns that uniquely identify a row (used to skip processed rows) |
| `MODELS_TO_USE` | one or more OpenRouter model ids; every row runs through every model |
| `BATCH_SIZE` | concurrent requests (`factcheck.py` only) |
| `DEBUG_BATCH_SIZE` | `0` = off; any other value runs only that many rows |

To score something other than accuracy, change `SYSTEM_PROMPT` and the
`Response` schema in the same file. `factcheck_batch.py` has an identical
settings block; keep the two in sync.
