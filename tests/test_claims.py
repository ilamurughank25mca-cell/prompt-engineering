import pytest

from agents import validate_claim_evidence
from models import (
    Claim,
    ClaimComparison,
    ClaimType,
    ComparisonCategory,
    Source,
    VerificationRecord,
    VerificationStatus,
    validate_claim_sources,
    validate_comparison_references,
    validate_verification_references,
)


def test_claim_source_id_must_exist_in_retrieved_sources():
    source = Source(
        source_id="S1",
        title="Article",
        url="https://example.com/article",
        publisher="Example News",
    )
    claim = Claim(
        claim_id="C1",
        text="The proposal passed.",
        classification=ClaimType.FACT,
        source_id="S2",
        evidence="The article says the proposal passed.",
    )

    with pytest.raises(ValueError, match="S2"):
        validate_claim_sources([claim], [source])


def test_claim_evidence_must_be_an_exact_source_excerpt():
    source = Source(
        source_id="S1",
        title="Article",
        url="https://example.com/article",
        publisher="Example News",
        content="The city council approved the measure on Tuesday.",
    )
    claim = Claim(
        claim_id="C1",
        text="The measure was approved.",
        classification=ClaimType.FACT,
        source_id="S1",
        evidence="The city council approved the measure on Tuesday.",
    )

    validate_claim_evidence([claim], [source])
    unsupported = claim.model_copy(update={"evidence": "The measure passed unanimously."})
    with pytest.raises(ValueError, match="exact excerpt"):
        validate_claim_evidence([unsupported], [source])


def test_comparison_rejects_unknown_claim_and_source_ids():
    source = Source(
        source_id="S1",
        title="Article",
        url="https://example.com/article",
        publisher="Example News",
    )
    comparison = ClaimComparison(
        comparison_id="CMP1",
        summary="The sources differ.",
        category=ComparisonCategory.CONFLICTING,
        claim_ids=["C9"],
        source_ids=["S8"],
        evidence_quality="Limited.",
        independence_note="Not established.",
    )

    with pytest.raises(ValueError, match="unknown claims: C9; unknown sources: S8"):
        validate_comparison_references([comparison], [], [source])


def test_verification_requires_cited_follow_up_evidence():
    source = Source(
        source_id="S2",
        title="Follow-up",
        url="https://example.com/follow-up",
        publisher="Example News",
        content="The signed order took effect on 12 May.",
    )
    claim = Claim(
        claim_id="C1",
        text="The order took effect in May.",
        classification=ClaimType.FACT,
        source_id="S1",
        evidence="The order took effect in May.",
    )
    record = VerificationRecord(
        claim_id="C1",
        query="order effective date",
        status=VerificationStatus.CORROBORATED,
        finding="A follow-up source supports the date.",
        source_ids=["S2"],
        evidence="The signed order took effect on 12 May.",
        iterations=1,
    )

    validate_verification_references([record], [claim], [source])
    unsupported = record.model_copy(update={"evidence": "The order took effect in June."})
    with pytest.raises(ValueError, match="not present"):
        validate_verification_references([unsupported], [claim], [source])