from evaluation import audit_citations, evaluate_analysis
from models import Claim, ClaimComparison, ClaimType, ComparisonCategory, Source


def test_audit_citations_reports_unknown_ids_and_missing_citations():
    source = Source(
        source_id="S1",
        title="Article",
        url="https://example.com/article",
        publisher="Example News",
    )

    assert audit_citations("A claim [S9].", [source]) == ["Unknown source IDs cited: S9."]
    assert audit_citations("No reference here.", [source]) == ["No source ID citations were detected."]


def test_audit_citations_accepts_known_source_id():
    source = Source(
        source_id="S1",
        title="Article",
        url="https://example.com/article",
        publisher="Example News",
    )

    assert audit_citations("A claim [S1].", [source]) == []


def test_evaluation_calculates_only_metrics_supported_by_inputs():
    source = Source(
        source_id="S1",
        title="Article",
        url="https://example.com/article",
        publisher="Example News",
        content="The council approved the plan.",
        search_perspectives=["background"],
    )
    claim = Claim(
        claim_id="C1",
        text="The council approved the plan.",
        classification=ClaimType.FACT,
        source_id="S1",
        evidence="The council approved the plan.",
    )
    comparison = ClaimComparison(
        comparison_id="CMP1",
        summary="Sources conflict.",
        category=ComparisonCategory.CONFLICTING,
        claim_ids=["C1"],
        source_ids=["S1"],
        evidence_quality="One article.",
        independence_note="Independent confirmation is not established.",
    )

    summary = evaluate_analysis(
        "Supported [S1], unknown [S8].",
        [claim],
        [comparison],
        [source],
        reference_claims=["The council approved the plan.", "A second reference claim."],
    )

    assert summary.claim_precision == 1.0
    assert summary.claim_recall == 0.5
    assert summary.citation_accuracy == 0.5
    assert summary.conflicting_comparisons == 1
    assert summary.unsupported_claims == 0
    assert summary.perspective_coverage == 0.25

    without_labels = evaluate_analysis("No citations.", [], [], [source])
    assert without_labels.claim_precision is None
    assert without_labels.claim_recall is None
    assert without_labels.citation_accuracy is None