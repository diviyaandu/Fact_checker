from factcheck.models import Claim
from factcheck.pipeline import get_claim_verdicts

claim_text = input("Enter claim: ").strip()

if not claim_text:
    raise SystemExit("Claim cannot be empty.")

claim = Claim(
    claim_id=1,
    claim_text=claim_text,
    start_time="00:00",
    end_time="00:00",
    category="General",
    importance="high",
    search_queries=[
        claim_text,
        f"{claim_text} fact check",
        f"is it true that {claim_text}"
    ]
)

print("\nRunning fact-checking pipeline...\n")

results = get_claim_verdicts(
    claims=[claim],
    use_cache=False
)

result = results[1]
verdict = result["verdict"]

print("\n========== FACT-CHECK RESULT ==========")
print("Claim:", claim_text)
print("Verdict:", verdict.verdict)
print("Confidence:", verdict.confidence)
print("Summary:", verdict.summary)
print("Explanation:", verdict.explanation)

print("\n========== SOURCES ==========")

for source in result["sources"]:
    print("\nTitle:", source.title)
    print("Publisher:", source.publisher)
    print("URL:", source.url)
    print("Evidence:", source.relevant_text)

print("\n========== CONFIDENCE BREAKDOWN ==========")
print(verdict.confidence_breakdown)