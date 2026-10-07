"""The one place that talks to Groq: a shared client, a logged chat call and
the small helpers for cleaning up the model's JSON output."""

import time

from groq import Groq

from factcheck import config
from factcheck.logging_utils import logger

client = Groq(api_key=config.GROQ_API_KEY)


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    stage: str,
    max_tokens: int,
    temperature: float = 0.1,
    log_fields: str = "",
) -> str:
    """One Groq chat completion; returns the stripped text.

    stage       short name used in the logs (e.g. "claim_extraction")
    log_fields  extra ' | key=value' text appended to the log lines
    """
    logger.info(
        "GROQ call | stage=%s | model=%s%s | input_chars=%d",
        stage, model, log_fields, len(user_prompt),
    )

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(
            "GROQ success | stage=%s%s | duration=%.2fs",
            stage, log_fields, time.time() - start,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        logger.exception("GROQ failure | stage=%s%s", stage, log_fields)
        raise


def strip_code_fences(text: str) -> str:
    """Removes ``` fences the model sometimes wraps around its JSON."""
    text = text.strip()

    if text.startswith("```"):
        lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
        text = "\n".join(lines)

    return text.strip()
