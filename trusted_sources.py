"""
trusted_sources.py

Domains are no longer used as a hard search filter — restricting search to a
tiny whitelist was silently starving the fact-checker of real evidence for
anything not covered by those exact domains (see Claims #1, #2, #4 in testing:
all UNVERIFIABLE purely because the true best source wasn't on the list).

Instead, we search the open web and score credibility AFTER retrieval, per
the project's Tier 1-5 source hierarchy (README Section 7). Untrusted/unknown
sources are still shown to the LLM — just ranked lowest and labeled clearly.
"""

# Tier 1 — Primary / official sources (government, international orgs, official stats)
TIER_1 = {
    "nasa.gov", "noaa.gov", "usgs.gov", "nsf.gov",
    "who.int", "cdc.gov", "nih.gov",
    "worldbank.org", "un.org", "loc.gov",
}

# Tier 2 — Scientific / academic
TIER_2 = {
    "nature.com", "sciencedirect.com", "britannica.com", "si.edu",
    "pubmed.ncbi.nlm.nih.gov", "arxiv.org",
}

# Tier 3 — Reputable news organizations
TIER_3 = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "ft.com", "aljazeera.com",
}

# Tier 4 — Fact-checking organizations + general reference
TIER_4 = {
    "snopes.com", "politifact.com", "factcheck.org",
    "en.wikipedia.org",
}

TIER_LABELS = {
    1: "Tier 1 · Government/Official",
    2: "Tier 2 · Scientific/Academic",
    3: "Tier 3 · Reputable News",
    4: "Tier 4 · Fact-Check/Reference",
    5: "Tier 5 · Unrated Source",
}


def score_credibility(publisher: str) -> int:
    """Lower number = higher credibility. Unknown sources get tier 5 (lowest),
    but are still returned — never silently dropped — so the LLM can decide
    whether they're usable, per Section 7's 'evaluate, don't hardcode' rule."""
    if publisher in TIER_1:
        return 1
    if publisher in TIER_2:
        return 2
    if publisher in TIER_3:
        return 3
    if publisher in TIER_4:
        return 4
    return 5


def label_for_tier(tier: int) -> str:
    return TIER_LABELS.get(tier, TIER_LABELS[5])