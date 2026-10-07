# YouTube Fact Checker

An AI-assisted fact-checking prototype. Paste a YouTube link and it:

1. gets the video's transcript,
2. picks out up to **5 checkable factual claims**,
3. searches the web for evidence on each claim,
4. asks an LLM to judge what each source says, and
5. returns a **verdict** and a **confidence score** for every claim, with the sources it relied on.

> **This is an academic prototype, not a production system.** Verdicts reflect the
> strength of the *retrieved evidence*, not absolute truth, and the confidence score
> is not a probability that the claim is true (see [Confidence score](#confidence-score)).

---

## How it works

```
YouTube URL
   │
   ▼
Transcript ──── official captions (youtube-transcript-api)
   │            └─ if none: download audio (yt-dlp) + local Whisper (faster-whisper)
   ▼
Claim extraction ──── Groq LLM → max 5 claims, each with 3 search queries
   │                  (timestamps come from the transcript, never from the LLM)
   ▼
Evidence search ──── Tavily, up to 3 queries per claim, run in parallel
   │                 de-duplicate by publisher → rank by credibility → max 5 sources/claim
   ▼
Verdicts ──── Groq LLM, 2 claims per call. For every source it returns structured
   │          judgments (relevance, stance, directness) and a verdict. No confidence number.
   ▼
Confidence ──── computed in Python from those judgments + source credibility
   │
   ▼
Streamlit UI ──── verdict, confidence bar, summary, explanation, cited sources
```

### Verdicts

| Verdict | Meaning |
|---|---|
| 🟢 **TRUE** | relevant evidence clearly confirms the claim as stated |
| 🔴 **FALSE** | relevant evidence clearly contradicts the claim |
| 🟡 **PARTIALLY TRUE** | some parts are accurate, but it overstates, omits context, or only holds under conditions |
| 🟠 **MISLEADING** | technically has some basis but creates a false impression |
| ⚪ **UNVERIFIABLE** | no relevant evidence, or not enough to judge — always confidence **0** |

### Guard rails on the LLM

- It may only cite sources that were actually retrieved for *that* claim. Citations whose URL
  doesn't match a retrieved source are dropped.
- A definite verdict with no valid citation is downgraded to **UNVERIFIABLE**.
- It never invents timestamps: it points at numbered transcript segments and the real
  start/end times are looked up from the transcript.
- If its JSON is malformed, the call is retried once with a stricter reminder; after that the
  claim is marked UNVERIFIABLE instead of crashing.

### When there are no sources

| Situation | Result |
|---|---|
| The search itself failed (technical problem) | UNVERIFIABLE, confidence 0 |
| Search worked but returned **nothing at all** (e.g. a name that looks invented or mis-transcribed) | FALSE with a fixed, low confidence of **0.45** — a deliberate heuristic, not produced by the formula |
| Search returned results but none were usable | UNVERIFIABLE, confidence 0 |

### Source credibility

Sources are scored **after** retrieval (the whole web is searched, nothing is whitelisted),
then ranked so the strongest evidence is shown to the LLM first. Unknown domains are kept but
rank lowest.

| Tier | Type | Examples | Weight in score |
|---|---|---|---|
| 1 | Government / official | nasa.gov, who.int, cdc.gov, nih.gov, un.org | 1.00 |
| 2 | Scientific / academic | nature.com, sciencedirect.com, britannica.com, arxiv.org | 0.85 |
| 3 | Reputable news | reuters.com, apnews.com, bbc.com, theguardian.com | 0.65 |
| 4 | Fact-check / reference | snopes.com, politifact.com, factcheck.org, wikipedia | 0.45 |
| 5 | Unrated | everything else | 0.25 |

The domain lists live in `factcheck/evidence/credibility.py`.

---

## Confidence score

The LLM judges *what each source says*; **Python computes the number**, so the same judgments
always give the same score and every part of it can be inspected.

For each source, the LLM returns (all on small fixed scales, which are more consistent than free-form decimals):

- **relevance** 0–3 — how directly it addresses the claim
- **stance** — `supports`, `contradicts`, `partial` or `neutral`
- **directness** 0–3 — how explicitly it states the specific fact

Each is normalised to 0–1 and combined with the source's credibility tier into a "mass"
`w = relevance × credibility × directness`. Only sources that are relevant, direct and
non-neutral count. Sources whose stance matches the verdict are the *aligned* ones
(TRUE → `supports`, FALSE → `contradicts`, PARTIALLY TRUE / MISLEADING → all counted sources).

Four components, each 0–1:

| Component | Weight | Definition |
|---|---|---|
| **Relevance** | 0.20 | mean relevance of the aligned sources |
| **Credibility** | 0.25 | relevance-weighted mean credibility of the aligned sources |
| **Strength** | 0.35 | *dominance* × mean directness, where dominance = aligned mass ÷ (aligned + opposing mass) |
| **Agreement** | 0.20 | independent aligned domains ÷ 3, capped at 1 |

```
confidence = 0.20·relevance + 0.25·credibility + 0.35·strength + 0.20·agreement
```

Edge cases, handled explicitly:

- **Conflicting evidence** — for TRUE/FALSE, if opposing evidence exists and dominance is
  below 0.75, confidence is **capped at 0.60**, however good the supporting sources are.
- **Never fully certain** — confidence is capped at **0.95**.
- **No usable evidence** — a TRUE/FALSE verdict with no aligned source is downgraded to UNVERIFIABLE (0).
- **UNVERIFIABLE** — always 0.
- Two sources from the same registrable domain (e.g. `news.bbc.co.uk` and `www.bbc.co.uk`)
  count as **one** independent source.

**Worked example** — three sources back a TRUE verdict: a Tier-1 source (relevance 3, directness 3),
a Tier-2 source (3, 2) and a Tier-3 source (2, 2), from three different domains:

| relevance | credibility | strength | agreement | **confidence** |
|---|---|---|---|---|
| 0.889 | 0.856 | 0.778 | 1.000 | **0.864** |

Each claim card in the app has a **"How was this confidence calculated?"** expander showing
these components, and the same numbers are written to the log (see [Logging](#logging)).

> The weights and thresholds are heuristic design choices, not fitted to data. They are all
> in one place, `factcheck/scoring/constants.py`, and are easy to tune. The score expresses how well
> the evidence supports **the verdict reached**, not how likely the claim is to be true.

---

## The app

- **Analyze Video** — runs the pipeline. Results stay on screen across Streamlit reruns.
- **Cooldown** — after an analysis that needs fresh API calls, the button is disabled for
  **60 seconds** (to protect free-tier Groq/Tavily limits). There's no countdown number: the
  button's normal colour fills **left → right** like a progress bar, and it re-enables when it
  is full. The cooldown survives reruns.
- **Cached analyses skip the cooldown** — if the transcript, claims and every claim's verdict
  are already cached, no API calls are needed, so the button is never locked for that video
  (even while another cooldown is running).
- **Use cached results** (sidebar) — turn it off for a clean re-run, e.g. to compare before/after
  a change to the verdict logic.
- **Clear cache** (sidebar) — deletes the cache; a "Cache cleared." notice fades away by itself.

### Caching

A small JSON file at `.cache/fact_checker_cache.json` stores three things:

| Namespace | Key | Contents |
|---|---|---|
| `transcript` | video ID | segments + how they were obtained |
| `claims` | video ID | the extracted claims |
| `claim_results` | hash of the claim text | sources, search metadata and the verdict (with its confidence breakdown) |

Caching speeds up demos and makes evaluation fair: you can re-run the same claims and evidence
while changing only one stage. Results cached before the confidence rewrite keep their old
Groq-assigned numbers until you clear the cache.

---

## Setup

**Requirements:** Python 3.10+, [ffmpeg](https://ffmpeg.org/) on your PATH (only used by the Whisper
fallback), a [Groq](https://console.groq.com/) API key and a [Tavily](https://tavily.com/) API key.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-key
TAVILY_API_KEY=your-tavily-key
```

Run it:

```bash
streamlit run app.py
```

(Needs Streamlit ≥ 1.39. The first Whisper fallback downloads the small `tiny` model.)

### Settings

| Setting | Where | Default | Notes |
|---|---|---|---|
| `GROQ_API_KEY`, `TAVILY_API_KEY` | `.env` | — | required |
| `GROQ_MODEL` | `.env` | `openai/gpt-oss-120b` | model used for verdicts (claim extraction is fixed to the same default) |
| `LOG_LEVEL` | `.env` | `INFO` | `DEBUG` also logs per-source scoring rows |
| `ANALYZE_COOLDOWN_SECONDS` | shell environment | `60` | `0` disables the cooldown. Set it in your terminal before launching, e.g. `ANALYZE_COOLDOWN_SECONDS=30 streamlit run app.py` (PowerShell: `$env:ANALYZE_COOLDOWN_SECONDS=30; streamlit run app.py`) |
| `MAX_CLAIMS` | `factcheck/config.py` | 5 | hard cap on claims per video |
| `MAX_RESULTS_PER_QUERY`, `MAX_SOURCES_PER_CLAIM`, `MAX_CONCURRENT_SEARCHES`, `MAX_SNIPPET_CHARS` | `factcheck/config.py` | 3 / 5 / 2 / 800 | evidence retrieval limits |
| `BATCH_SIZE` | `factcheck/config.py` | 2 | claims per Groq verdict call |
| `WHISPER_MODEL_SIZE` | `factcheck/config.py` | `tiny` | speech-to-text model |
| scoring weights, caps | `factcheck/scoring/constants.py` | see above | |
| UI cooldown / notice timing | `ui/config.py` | 60 s / 3.5 s | |

---

## Logging

Every stage logs a one-line, greppable record (`CACHE hit/miss`, `GROQ call/success/failure`,
`TAVILY …`, `SEARCH complete`, `BATCH start/complete`, `VERDICT`, …). Confidence adds one line per claim:

```
CONFIDENCE | claim_id=1 | relevance=0.89 | credibility=0.86 | strength=0.78 | agreement=1.00 | dominance=1.00 | weighted_sum=0.86 | conflict_capped=False | final=0.86
```

---

## Project layout

```
app.py                  Streamlit entry point (a few lines)

factcheck/              the whole backend (no Streamlit anywhere in here)
  config.py               every backend setting: keys, models, limits, batch size
  models.py               Claim, Source, CitedSource, Verdict
  llm.py                  the one Groq client + logged chat call
  cache.py                JSON disk cache (.cache/ in the project root)
  logging_utils.py        the shared logger
  transcript/             getting the transcript
    segments.py             segment format shared by every source
    youtube.py              URL validation, video ID, official captions
    whisper.py              audio download + local Whisper fallback
  claims/                 claim extraction
    prompt.py               the extraction prompt
    extractor.py            extract_claims + validation of each proposed claim
  evidence/               evidence retrieval
    credibility.py          Tier 1-5 domain lists
    search.py               Tavily queries, parallel run, de-dup, ranking
  verdicts/               LLM verdicts
    prompt.py               the verdict prompt + the claim/source text builder
    checker.py              one Groq call per batch, with one retry
    parsing.py              JSON -> Verdict: real citations only, safe downgrades
    no_sources.py           verdicts when search found nothing (no LLM call)
  scoring/                deterministic confidence
    constants.py            weights, tier credibility, caps
    assessments.py          the LLM's per-source judgments + validation
    confidence.py           the formula
  pipeline/               orchestration + caching - the only API the UI uses
    transcripts.py          transcript cache -> captions -> Whisper
    claims.py               claims cache / extract / save
    verdicts.py             evidence search + batching + caching
    recording.py            logging and caching of each finished verdict
    cache_check.py          "is this whole video already cached?"

ui/                     the front end, one concern per file
  config.py               cooldown length, notice duration, widget keys
  timing.py               pure cooldown arithmetic
  styles.py               CSS for the filling button and the fading notice
  state.py                everything kept in st.session_state
  sidebar.py              cache controls
  controls.py             URL box, Analyze button, cooldown logic
  analysis.py             runs the pipeline for one URL
  results.py              draws the results

tests/                  no network, no API keys needed
```

**Import direction is one-way:** `pipeline` -> `transcript`, `claims`, `evidence`, `verdicts` ->
`scoring`, `llm`, `models`, `config`, `cache`, `logging_utils`. Nothing imports upward, and the UI
only talks to `factcheck.pipeline` (plus the URL helpers in `factcheck.transcript`).

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests need no API keys or network access - Groq, Tavily, YouTube and Streamlit are replaced
with fakes.

| File | Covers |
|---|---|
| `test_scoring.py` | the confidence formula (hand-checked), conflict cap, edge cases, and the batched pipeline end to end with caching |
| `test_claims.py` | claim validation, real timestamps, the 5-claim cap, query fallbacks, JSON retry |
| `test_evidence.py` | de-duplication, caps, ranking, metadata, concurrency limit, failure handling |
| `test_verdicts.py` | prompt layout, citation validation, batching, retry, reordering |
| `test_transcript.py` | URL parsing, segment format, official captions |
| `test_cache_check.py` | the "fully cached" check behind the cooldown bypass |
| `test_ui.py` | cooldown lifecycle, cached bypass, fading notice |

## Limitations

- Evidence comes from **search-result snippets** (up to 800 characters), not whole pages.
- The LLM's per-source judgments are consistent but not perfectly deterministic (low temperature,
  not zero); only the arithmetic on top of them is.
- Claim extraction depends on transcript quality; Whisper's `tiny` model trades accuracy for speed.
- Maximum 5 claims per video, by design.
- Heuristic weights and credibility tiers reflect design judgment, not empirical calibration.
