# News Analysis Agent

A staged academic project for researching a news topic with Tavily and drafting a source-cited analysis with OpenAI. The current minimum version retrieves and deduplicates news results, assigns source IDs, creates a report using one of three prompt strategies, and audits citations against the retrieved source IDs.

## Files and responsibilities

- `app.py`: Streamlit controls, progress, error display, source list, report, and Markdown download.
- `retrieval.py`: Tavily search, URL normalization, duplicate removal, and source records.
- `agents.py`: OpenAI report synthesis with bounded source text and untrusted-content instructions.
- `prompts.py`: Basic, role-based, and structured evidence-based prompt strategies.
- `models.py`: Pydantic source and claim schemas plus claim/source ID validation.
- `evaluation.py`: Citation ID audit. Broader extraction metrics are planned for a later stage.
- `tests/`: focused tests for retrieval normalization, claim source validation, and citations.
- `requirements.txt`: runtime and test dependencies.
- `.env.example`: API key placeholders; copy to `.env` locally.

## Current workflow

1. Enter a topic and choose a result count, recency window, and prompt strategy.
2. Tavily retrieves news results; invalid and duplicate URLs are discarded and remaining sources receive IDs such as `[S1]`.
3. The selected prompt strategy and the same source dataset are sent to the model. Article text is treated as untrusted input and capped per source.
4. The app displays source links and the report, warns about unknown or missing source ID citations, and offers a Markdown download.

This is the first working slice, not the complete multi-agent design. Claim extraction, comparison, targeted verification, and final-review agents are subsequent implementation stages. The citation audit checks whether an ID exists; it does not prove that the article supports the statement.

## Setup on Windows (PowerShell)

From the parent directory of this project:

```powershell
mkdir news-analysis-agent
cd news-analysis-agent
```

If you already opened the project folder in VS Code, start at virtual environment creation:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your own Tavily and OpenAI keys in `.env`. Do not commit that file. For OpenRouter, set
`OPENAI_BASE_URL=https://openrouter.ai/api/v1` and set `OPENAI_MODEL` to an available
OpenRouter model ID (for example, `openai/gpt-4o-mini`). Use a newly generated credential;
never paste API keys into chat or commit them. Then run:

```powershell
streamlit run app.py
```

Run automated tests from the project directory:

```powershell
pytest -q
```

If PowerShell blocks activation, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\python.exe -m streamlit run app.py` without activating the environment.

## Prompt experiment and interpretation

The three strategies operate on the same retrieved source set within an analysis, so the instruction style is the main changed variable. The current evaluation reports citation-ID integrity only. It does not calculate claim precision, recall, citation support accuracy, unsupported-claim rates, conflict detection, or perspective coverage; those require extracted claims and human-labeled reference examples. Do not treat an absent score as a zero or as evidence of quality.

## Limitations and next stages

Search coverage depends on Tavily, model output can still be incomplete or wrong, publisher identity is inferred from the result URL, and publication dates may be absent. The app does not yet perform structured claim extraction or targeted verification and does not independently verify source credibility. Next, add structured claim extraction and validation, then claim comparison, bounded verification, report review, and labeled evaluation.