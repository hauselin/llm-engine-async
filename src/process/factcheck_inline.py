"""
Fact-check a handful of inline texts. No input file, no load_and_prepare.
output: printed as JSON lines (set OUTFILE to write to data/processed instead)
run: uv run python src/process/factcheck_inline.py
"""

# %% Setup

import asyncio

import httpx
from pydantic import BaseModel, Field, field_validator

from src.llm_engine import ModelResponse, openrouter_chat, run_batch, setup_env

setup_env()

# %% Parameters

MODELS_TO_USE = ["google/gemini-3.1-flash-lite"]
BATCH_SIZE = 10

TEXTS = [
    {"content": "The Eiffel Tower is in Paris and was completed in 1889."},
    {"content": "Water boils at 50 degrees Celsius at sea level."},
    {"content": "I think pineapple on pizza is the best topping."},
]

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


# %% Build tasks: every text x every model

dat = [row | {"model": model} for model in MODELS_TO_USE for row in TEXTS]
assert len(dat) == len(TEXTS) * len(MODELS_TO_USE)

# %% Async LLM call


async def get_llm_response(
    semaphore: asyncio.Semaphore, data: dict, client: httpx.AsyncClient
) -> tuple[ModelResponse | Exception, dict, float]:
    async with semaphore:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data["content"].strip()},
        ]
        try:
            response = await openrouter_chat(
                client,
                model=data["model"],
                messages=messages,
                response_model=Response,
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
