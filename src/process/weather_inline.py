"""
Ask for today's weather in three cities via OpenRouter's web_search tool.
Tests that the tool is honoured: the model cannot know current weather otherwise.
output: printed as JSON lines (set OUTFILE to write to data/processed instead)
run: uv run python src/process/weather_inline.py
"""

# %% Setup

import asyncio
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from src.llm_engine import ModelResponse, openrouter_chat, run_batch, setup_env

setup_env()

# %% Parameters

MODELS_TO_USE = ["google/gemini-3.1-flash-lite"]

OUT_DIR = Path("data/processed")
OUTFILE: str | None = None  # e.g. "weather.jsonl"; None prints results

BATCH_SIZE = 10
N_RETRIES = 3

CITIES = [
    {"id": 1, "city": "Toronto", "country": "Canada"},
    {"id": 2, "city": "Tokyo", "country": "Japan"},
    {"id": 3, "city": "Cape Town", "country": "South Africa"},
]

SYSTEM_PROMPT = """
You report current weather. Use web search to find today's conditions for the requested city; do not guess from memory. Respond with valid JSON matching the schema provided, and cite the source URL you relied on.
"""

# %% Response schema for LLM structured output


class Response(BaseModel):
    temperature_c: float = Field(description="Current temperature in degrees Celsius")
    conditions: str = Field(description="Short description, e.g. 'light rain'")
    observed_at: str = Field(
        description="Date/time of the observation as reported by the source"
    )
    source_url: str = Field(description="URL of the page the weather was taken from")


# %% Build tasks: every city x every model

dat = [row | {"model": model} for model in MODELS_TO_USE for row in CITIES]
assert len(dat) == len(CITIES) * len(MODELS_TO_USE)

# %% Async LLM call


async def get_llm_response(
    semaphore: asyncio.Semaphore, data: dict, client: httpx.AsyncClient
) -> tuple[ModelResponse | Exception, dict, float]:
    async with semaphore:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"What is the weather right now in {data['city']}, {data['country']}?",
            },
        ]
        try:
            response = await openrouter_chat(
                client,
                model=data["model"],
                messages=messages,
                response_model=Response,
                tools=[
                    {
                        "type": "openrouter:web_search",
                        "parameters": {"engine": "native"},
                    }
                ],
            )
            return response, data, response.usage.cost
        except Exception as e:  # noqa: BLE001
            return e, data, 0


# %%

asyncio.run(
    run_batch(
        dat,
        get_llm_response,
        Response,
        batch_size=BATCH_SIZE,
    )
)
