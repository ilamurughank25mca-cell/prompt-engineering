import re
import string

from models import (
    Claim,
    ClaimComparison,
    ComparisonCategory,
    EvaluationSummary,
    SearchPerspective,
    Source,
)


def audit_citations(report: str, sources: list[Source]) -> list[str]:
    cited_ids = set(re.findall(r"\[(S\d+)\]", report))
    known_ids = {source.source_id for source in sources}
    issues = []
    unknown_ids = sorted(cited_ids - known_ids)
    if unknown_ids:
        issues.append(f"Unknown source IDs cited: {', '.join(unknown_ids)}.")
    if not cited_ids:
        issues.append("No source ID citations were detected.")
    return issues


def evaluate_analysis(
    report: str,
    claims: list[Claim],
    comparisons: list[ClaimComparison],
    sources: list[Source],
    reference_claims: list[str] | None = None,
) -> EvaluationSummary:
    cited_ids = re.findall(r"\[(S\d+)\]", report)
    known_ids = {source.source_id for source in sources}
    citation_accuracy = (
        sum(source_id in known_ids for source_id in cited_ids) / len(cited_ids)
        if cited_ids
        else None
    )
    source_by_id = {source.source_id: source for source in sources}
    unsupported_claims = 0
    for claim in claims:
        source = source_by_id.get(claim.source_id)
        evidence = " ".join(claim.evidence.casefold().split())
        content = " ".join(source.content.casefold().split()) if source else ""
        if not evidence or evidence not in content:
            unsupported_claims += 1

    precision = recall = None
    if reference_claims is not None:
        normalize = lambda value: " ".join(
            "".join(character for character in value.casefold() if character not in string.punctuation).split()
        )
        extracted = {normalize(claim.text) for claim in claims}
        reference = {normalize(text) for text in reference_claims}
        overlap = len(extracted & reference)
        precision = overlap / len(extracted) if extracted else None
        recall = overlap / len(reference) if reference else None

    expected_perspectives = {perspective.value for perspective in SearchPerspective}
    covered = sorted({perspective for source in sources for perspective in source.search_perspectives})
    return EvaluationSummary(
        claim_count=len(claims),
        claim_precision=precision,
        claim_recall=recall,
        citation_accuracy=citation_accuracy,
        conflicting_comparisons=sum(item.category == ComparisonCategory.CONFLICTING for item in comparisons),
        unsupported_claims=unsupported_claims,
        covered_perspectives=covered,
        perspective_coverage=len(set(covered) & expected_perspectives) / len(expected_perspectives),
    )