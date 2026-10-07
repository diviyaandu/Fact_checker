"""The fact-checking backend (everything that isn't Streamlit).

Data flows through these sub-packages in order:

    transcript/  YouTube URL handling, official captions, Whisper fallback
    claims/      LLM extracts up to MAX_CLAIMS checkable claims (+ search queries)
    evidence/    Tavily web search, de-duplication, source-credibility tiers
    verdicts/    LLM judges each source and picks a verdict (no confidence number)
    scoring/     deterministic confidence from the LLM's per-source judgments
    pipeline/    orchestration + caching; the only API the UI needs

Shared plumbing lives at this level:

    config.py        every tunable constant / environment variable
    models.py        Claim, Source, CitedSource, Verdict (pydantic)
    llm.py           the Groq client and the one place that calls it
    cache.py         tiny JSON disk cache (.cache/ in the project root)
    logging_utils.py the "fact_checker" logger

Import direction is one-way: pipeline -> {transcript, claims, evidence,
verdicts} -> {scoring, llm, models, config, cache, logging_utils}.
"""
