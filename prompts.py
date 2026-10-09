PROMPT_STRATEGIES = {
    "Basic": (
        "Summarize the supplied news sources about the topic in clear, concise language. "
        "Mention where the sources agree or differ."
    ),
    "Role-based": (
        "Act as a neutral news analyst. Compare the supplied sources, represent differing "
        "perspectives fairly, and distinguish reported information from interpretation."
    ),
    "Structured evidence-based": (
        "Prepare a structured analytical report with an executive summary, background, key "
        "claims and evidence, areas of agreement, conflicts, source limitations, evidence gaps, "
        "and a cautious assessment. Support factual claims with source IDs and explicitly mark "
        "uncertainty. Do not treat repeated reporting as independent confirmation."
    ),
}