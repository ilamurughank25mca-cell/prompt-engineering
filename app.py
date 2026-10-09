import os

import streamlit as st
from dotenv import load_dotenv

from agents import (
    AgentError,
    compare_claims,
    extract_claims,
    plan_research,
    review_report,
    synthesize_report,
    verify_claims,
)
from evaluation import audit_citations, evaluate_analysis
from prompts import PROMPT_STRATEGIES
from retrieval import RetrievalError, search_queries


load_dotenv()
st.set_page_config(page_title="News Analysis Agent", page_icon="N", layout="wide")
st.title("News Analysis Agent")
st.caption("Multi-source research, evidence checks, and balanced reporting.")

with st.form("analysis_form"):
    topic = st.text_input("News topic", placeholder="For example: recent changes to urban air quality rules")
    col1, col2, col3 = st.columns(3)
    with col1:
        result_count = st.slider("Maximum unique sources", min_value=3, max_value=10, value=6)
    with col2:
        recency_days = st.number_input("Limit to recent days (0 = no limit)", min_value=0, max_value=365, value=30)
    with col3:
        strategy = st.selectbox("Prompt strategy", list(PROMPT_STRATEGIES))
    compare_strategies = st.checkbox("Compare all prompt strategies on the same retrieved dataset")
    reference_text = st.text_area(
        "Optional reference claims (one per line)",
        help="Used only for exact-match precision and recall. Leave blank when no labeled reference set is available.",
    )
    submitted = st.form_submit_button("Analyze News", type="primary")

if submitted:
    if not topic.strip():
        st.error("Enter a news topic to analyze.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("Add your OpenAI API key as OPENAI_API_KEY in .env, then restart Streamlit.")
    elif not os.getenv("TAVILY_API_KEY"):
        st.error("Add TAVILY_API_KEY for web retrieval. An OpenAI API key does not include Tavily search access.")
    else:
        try:
            progress = st.progress(0, text="Starting research")
            with st.status("Running the research agents", expanded=True) as status:
                status.write("Research planning: generating balanced search queries.")
                plan = plan_research(topic.strip(), os.environ["OPENAI_API_KEY"])
                progress.progress(12, text="Research plan ready")

                status.write(f"Source retrieval: searching {len(plan.queries)} perspectives.")
                sources = search_queries(
                    plan,
                    os.environ["TAVILY_API_KEY"],
                    max_results=result_count,
                    days=recency_days or None,
                )
                progress.progress(25, text="Source retrieval complete")
                if not sources:
                    status.update(label="No sources found", state="error")
                    st.warning("No usable articles were returned. Broaden the topic or date range and try again.")
                else:
                    status.write("Claim extraction: structuring claims and checking exact excerpts.")
                    claims = extract_claims(sources, os.environ["OPENAI_API_KEY"]).claims
                    progress.progress(42, text="Claims extracted and source-checked")

                    status.write("Claim comparison: checking agreement and conflicts.")
                    comparisons = compare_claims(claims, sources, os.environ["OPENAI_API_KEY"]).comparisons
                    progress.progress(57, text="Claims compared")

                    status.write("Verification: targeted searches for unresolved or disputed claims.")
                    verifications, sources = verify_claims(
                        claims,
                        comparisons,
                        sources,
                        os.environ["TAVILY_API_KEY"],
                        os.environ["OPENAI_API_KEY"],
                        max_iterations=1,
                    )
                    progress.progress(70, text="Verification pass complete")

                    strategies = list(PROMPT_STRATEGIES) if compare_strategies else [strategy]
                    reports = {}
                    for index, prompt_strategy in enumerate(strategies):
                        status.write(f"Report synthesis: applying the {prompt_strategy} strategy.")
                        reports[prompt_strategy] = synthesize_report(
                            topic.strip(),
                            sources,
                            os.environ["OPENAI_API_KEY"],
                            prompt_strategy,
                            claims,
                            comparisons,
                            verifications,
                        )
                        progress.progress(72 + int(18 * (index + 1) / len(strategies)), text="Drafting report")

                    report = reports[strategy]
                    status.write("Final review: checking citation IDs and evidence limitations.")
                    review = review_report(report, claims, sources, os.environ["OPENAI_API_KEY"])
                    progress.progress(100, text="Analysis complete")
                    status.update(label="Analysis complete", state="complete")

            if sources:
                missing_perspectives = sorted(
                    {query.perspective.value for query in plan.queries}
                    - {perspective for source in sources for perspective in source.search_perspectives}
                )
                if missing_perspectives:
                    st.warning("No unique sources surfaced for these perspectives: " + ", ".join(missing_perspectives))

                st.subheader("Retrieved sources")
                for source in sources:
                    published = f" · {source.published_at}" if source.published_at else " · date unavailable"
                    duplicate_note = f" · possible related reporting ({source.possible_duplicate_group})" if source.possible_duplicate_group else ""
                    perspectives = ", ".join(source.search_perspectives) or "unclassified"
                    st.markdown(
                        f"**[{source.source_id}] {source.title}** · {source.publisher}{published}{duplicate_note}  \n"
                        f"Perspectives: {perspectives} · [Open article]({source.url})"
                    )

                st.subheader("Extracted claims")
                if claims:
                    st.dataframe(
                        [
                            {
                                "Claim ID": claim.claim_id,
                                "Claim": claim.text,
                                "Classification": claim.classification.value,
                                "Source": claim.source_id,
                                "Evidence excerpt": claim.evidence,
                                "Uncertainty": claim.uncertainty or "Not stated",
                            }
                            for claim in claims
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No claims were extracted from the retrieved text.")

                st.subheader("Claim comparison")
                if comparisons:
                    st.dataframe(
                        [
                            {
                                "Category": item.category.value,
                                "Comparison": item.summary,
                                "Claims": ", ".join(item.claim_ids),
                                "Sources": ", ".join(item.source_ids),
                                "Evidence quality": item.evidence_quality,
                                "Independence note": item.independence_note,
                            }
                            for item in comparisons
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No comparisons were returned.")

                st.subheader("Verification results")
                if verifications:
                    st.dataframe(
                        [
                            {
                                "Claim": record.claim_id,
                                "Status": record.status.value,
                                "Finding": record.finding,
                                "Follow-up sources": ", ".join(record.source_ids) or "None",
                                "Search iterations": record.iterations,
                            }
                            for record in verifications
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No disputed or insufficiently supported claims required targeted verification.")

                st.subheader("Final review")
                if review.passed:
                    st.success("The report passed the structured review checks.")
                else:
                    st.warning("The reviewer flagged limitations or possible evidence gaps.")
                if review.requires_additional_research:
                    st.warning("The final review recommends additional research.")
                for finding in review.findings:
                    related = f" ({', '.join(finding.related_source_ids)})" if finding.related_source_ids else ""
                    st.write(f"**{finding.severity}**: {finding.issue}{related}")

                citation_issues = audit_citations(report, sources)
                if citation_issues:
                    st.warning("Citation ID audit: " + " ".join(citation_issues))

                reference_claims = [line.strip() for line in reference_text.splitlines() if line.strip()] or None
                metrics = evaluate_analysis(report, claims, comparisons, sources, reference_claims)
                st.subheader("Evaluation")
                metric_columns = st.columns(4)
                metric_columns[0].metric("Claims extracted", metrics.claim_count)
                metric_columns[1].metric("Claim precision", "N/A" if metrics.claim_precision is None else f"{metrics.claim_precision:.0%}")
                metric_columns[2].metric("Claim recall", "N/A" if metrics.claim_recall is None else f"{metrics.claim_recall:.0%}")
                metric_columns[3].metric("Citation ID accuracy", "N/A" if metrics.citation_accuracy is None else f"{metrics.citation_accuracy:.0%}")
                st.write(
                    f"Conflicting comparisons: {metrics.conflicting_comparisons} · "
                    f"Unsupported extracted claims: {metrics.unsupported_claims} · "
                    f"Perspective coverage: {metrics.perspective_coverage:.0%} "
                    f"({', '.join(metrics.covered_perspectives) or 'none'})"
                )
                st.caption(
                    "Claim precision/recall use exact normalized text matching against the optional labels. "
                    "Citation accuracy checks source ID existence only, not whether the source substantively proves a sentence."
                )

                st.subheader("Analytical report")
                if compare_strategies:
                    report_tabs = st.tabs(strategies)
                    for report_tab, prompt_strategy in zip(report_tabs, strategies):
                        with report_tab:
                            st.markdown(reports[prompt_strategy])
                else:
                    st.markdown(report)
                markdown_report = f"# News Analysis: {topic.strip()}\n\n{report}\n\n## References\n"
                markdown_report += "\n".join(
                    f"- [{source.source_id}] [{source.title}]({source.url}) — {source.publisher}"
                    + (f", {source.published_at}" if source.published_at else "")
                    for source in sources
                )
                st.download_button(
                    "Download Markdown report",
                    data=markdown_report,
                    file_name="news-analysis-report.md",
                    mime="text/markdown",
                )
        except RetrievalError as exc:
            st.error(f"Source retrieval failed: {exc}")
        except AgentError as exc:
            st.error(f"An agent could not complete its step: {exc}")