"""Backend for OpenRouter API"""

import json
import logging
import os
import time

from funcy import notnone, once, select_values
import openai

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    backoff_create,
)

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    # Handle structured output via OpenRouter's tool calling support (matching OpenAI backend)
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # Force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
        logger.info(f"Using OpenRouter tool calling for {func_spec.name}")

    # in case some backends dont support system roles, just convert everything to user
    messages = [
        {"role": "user", "content": message}
        for message in [system_message, user_message]
        if message
    ]

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body={
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    # Parse response based on whether tool calling was used
    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"tool_calls is empty, not a tool call response: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), f"Function name mismatch: expected {func_spec.name}, got {choice.message.tool_calls[0].function.name}"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
            logger.info(f"Successfully parsed tool call response for {func_spec.name}")
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
