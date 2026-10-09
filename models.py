from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class ClaimType(str, Enum):
    FACT = "factual statement"
    OPINION = "opinion"
    ALLEGATION = "allegation"
    PREDICTION = "prediction"


class Source(BaseModel):
    source_id: str
    title: str
    url: HttpUrl
    publisher: str
    published_at: str | None = None
    content: str = ""
    search_perspectives: list[str] = Field(default_factory=list)
    possible_duplicate_group: str | None = None


class SearchPerspective(str, Enum):
    BACKGROUND = "background"
    PRIMARY_SOURCES = "primary sources"
    OPPOSING_VIEWS = "opposing viewpoints"
    RECENT_REPORTING = "recent reporting"


class SearchQuery(BaseModel):
    query: str = Field(min_length=3, max_length=240)
    perspective: SearchPerspective


class ResearchPlan(BaseModel):
    queries: list[SearchQuery] = Field(min_length=1, max_length=8)


class ComparisonCategory(str, Enum):
    CORROBORATED = "corroborated"
    CONFLICTING = "conflicting"
    SINGLE_SOURCE = "single-source"
    INSUFFICIENT = "insufficient evidence"


class ClaimComparison(BaseModel):
    comparison_id: str
    summary: str
    category: ComparisonCategory
    claim_ids: list[str] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    evidence_quality: str
    independence_note: str


class ComparisonResult(BaseModel):
    comparisons: list[ClaimComparison] = Field(max_length=40)


class VerificationStatus(str, Enum):
    CORROBORATED = "corroborated"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


class VerificationRecord(BaseModel):
    claim_id: str
    query: str
    status: VerificationStatus
    finding: str
    source_ids: list[str] = Field(default_factory=list)
    evidence: str | None = None
    iterations: int = Field(ge=0, le=2)


class VerificationCollection(BaseModel):
    verifications: list[VerificationRecord] = Field(max_length=12)


class ReviewFinding(BaseModel):
    severity: str
    issue: str
    related_source_ids: list[str] = Field(default_factory=list)


class FinalReview(BaseModel):
    passed: bool
    findings: list[ReviewFinding] = Field(default_factory=list)
    requires_additional_research: bool = False


class EvaluationSummary(BaseModel):
    claim_count: int
    claim_precision: float | None = None
    claim_recall: float | None = None
    citation_accuracy: float | None = None
    conflicting_comparisons: int
    unsupported_claims: int
    covered_perspectives: list[str]
    perspective_coverage: float


class Claim(BaseModel):
    claim_id: str
    text: str
    classification: ClaimType
    source_id: str
    evidence: str
    uncertainty: str | None = None


class ClaimCollection(BaseModel):
    claims: list[Claim] = Field(max_length=40)


def validate_claim_sources(claims: list[Claim], sources: list[Source]) -> None:
    source_ids = {source.source_id for source in sources}
    unknown_ids = sorted({claim.source_id for claim in claims} - source_ids)
    if unknown_ids:
        raise ValueError(f"Claims reference unknown source IDs: {', '.join(unknown_ids)}")


def validate_comparison_references(
    comparisons: list[ClaimComparison], claims: list[Claim], sources: list[Source]
) -> None:
    claim_ids = {claim.claim_id for claim in claims}
    source_ids = {source.source_id for source in sources}
    unknown_claim_ids = {claim_id for item in comparisons for claim_id in item.claim_ids} - claim_ids
    unknown_source_ids = {source_id for item in comparisons for source_id in item.source_ids} - source_ids
    if unknown_claim_ids or unknown_source_ids:
        details = []
        if unknown_claim_ids:
            details.append(f"unknown claims: {', '.join(sorted(unknown_claim_ids))}")
        if unknown_source_ids:
            details.append(f"unknown sources: {', '.join(sorted(unknown_source_ids))}")
        raise ValueError("Comparison references " + "; ".join(details))


def validate_verification_sources(records: list[VerificationRecord], sources: list[Source]) -> None:
    source_ids = {source.source_id for source in sources}
    unknown_ids = {source_id for record in records for source_id in record.source_ids} - source_ids
    if unknown_ids:
        raise ValueError(f"Verification references unknown source IDs: {', '.join(sorted(unknown_ids))}")


def validate_verification_references(
    records: list[VerificationRecord], claims: list[Claim], sources: list[Source]
) -> None:
    claim_ids = {claim.claim_id for claim in claims}
    unknown_claim_ids = {record.claim_id for record in records} - claim_ids
    if unknown_claim_ids:
        raise ValueError(f"Verification references unknown claims: {', '.join(sorted(unknown_claim_ids))}")
    validate_verification_sources(records, sources)
    source_by_id = {source.source_id: source for source in sources}
    for record in records:
        if record.status != VerificationStatus.UNRESOLVED:
            evidence = " ".join((record.evidence or "").casefold().split())
            cited_text = " ".join(
                " ".join(source_by_id[source_id].content.casefold().split())
                for source_id in record.source_ids
            )
            if not evidence or evidence not in cited_text:
                raise ValueError(f"Verification evidence for {record.claim_id} is not present in its cited sources.")


def validate_review_sources(review: FinalReview, sources: list[Source]) -> None:
    source_ids = {source.source_id for source in sources}
    unknown_ids = {
        source_id
        for finding in review.findings
        for source_id in finding.related_source_ids
    } - source_ids
    if unknown_ids:
        raise ValueError(f"Review references unknown source IDs: {', '.join(sorted(unknown_ids))}")