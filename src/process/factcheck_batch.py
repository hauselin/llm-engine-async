"""
Same task as factcheck.py via OpenRouter's Batch API (50% price, results within 24h).
Each run collects finished batches into OUTFILE, then submits rows not yet in flight.
Run it once to submit, and again later (up to 24h) to collect.

input: data/processed/example-data.csv
output: data/processed/example-data-factcheck.jsonl (+ example-data-factcheck.jsonl.batches.json)
run: uv run python src/process/factcheck_batch.py
"""

# %% Setup

from pathlib import Path

import polars as pl
from polars import col as c
from pydantic import BaseModel, Field, field_validator

from src.llm_engine import load_and_prepare, run_batch_api, setup_env

setup_env()

# %% Define paths and parameters

MODELS_TO_USE = ["google/gemini-3.1-flash-lite"]

DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR

INPUT_FILE = "example-data.csv"
CONTENT_COL = "content"
OUTFILE = "example-data-factcheck.jsonl"

# NOTE: update ID_COLS to match the input data columns accordingly
ID_COLS = ["responseid", "model", "id"]

DEBUG_BATCH_SIZE = 0  # > 0 processes only that many rows (debug mode)

SYSTEM_PROMPT = """
<role>
You are a professional fact-checker, and you excel at two things: 1) evaluating whether a text contains mostly opinion or mostly factual claims, and 2) objectively evaluating the veracity/accuracy of information.
</role>

<task>
<task_1 field="has_factual_content">
You must first determine whether the text contains factual information to evaluate for accuracy using a response of either 0 (does not contain any factual information at all) or 1 (contains at least some factual information that can be evaluated for accuracy).
</task_1>

<task_2 field="accuracy_score">
You must then rate the overall veracity of the text using a scale of 0 (completely inaccurate) to 100 (completely accurate).
</task_2>

<task_3 field="explanation">
Provide a brief explanation of your response/reasoning for the factual content evaluation and accuracy score.
</task_3>
</task>

<output_format>
You must respond with a valid JSON matching the schema provided, where has_factual_content is the score for the first task, accuracy_score is the score for the second task, and explanation is a brief explanation of your response/reasoning.
</output_format>
"""

# %% Response schema for LLM structured output


class Response(BaseModel):
    has_factual_content: int = Field(
        description="Binary: 0 if no reasoning/logic/factual claims, 1 if contains factual or logical information that can be evaluated for accuracy",
    )
    accuracy_score: int = Field(
        description="Accuracy rating: 0 = completely inaccurate, 100 = completely accurate",
    )
    explanation: str = Field(
        description="Brief explanation (1-2 sentences) of the factual content evaluation and accuracy score.",
    )

    @field_validator("has_factual_content")
    @classmethod
    def check_has_factual_content(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"has_factual_content must be 0 or 1, got {v}")
        return v

    @field_validator("accuracy_score")
    @classmethod
    def check_accuracy_score(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError(f"accuracy_score must be between 0 and 100, got {v}")
        return v


# %% Load data, cross-join with models, drop already-processed rows


def score_ranges(d: pl.DataFrame) -> pl.DataFrame:
    """llm_engine derives only is_not_null for int fields; enforce the ranges too."""
    return d.filter(c("has_factual_content").is_between(0, 1)).filter(
        c("accuracy_score").is_between(0, 100)
    )


dat = load_and_prepare(
    DATA_DIR,
    INPUT_FILE,
    OUTFILE,
    MODELS_TO_USE,
    ID_COLS,
    Response,
    extra_filters=score_ranges,
    out_dir=OUT_DIR,
)

# %% Request body per row (model and response_format are added by the engine)


def build_request(data: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data[CONTENT_COL].strip()},
        ],
    }


# %%

run_batch_api(
    dat,
    build_request,
    Response,
    OUT_DIR,
    OUTFILE,
    ID_COLS,
    debug_batch_size=DEBUG_BATCH_SIZE,
)
