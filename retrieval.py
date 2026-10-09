import re
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tavily import TavilyClient

from models import ResearchPlan, Source


class RetrievalError(RuntimeError):
    """Raised when source search fails or returns unusable data."""


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    ignored_params = {"fbclid", "gclid"}
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query) if not key.lower().startswith("utm_") and key.lower() not in ignored_params]
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def normalize_results(results: list[dict], perspective: str | None = None) -> list[Source]:
    sources = []
    seen_urls = set()
    for result in results:
        url = result.get("url", "")
        canonical_url = _canonical_url(url) if isinstance(url, str) else ""
        if not canonical_url or canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)
        publisher = result.get("publisher") or urlsplit(canonical_url).netloc.removeprefix("www.")
        content = result.get("raw_content") or result.get("content") or ""
        sources.append(
            Source(
                source_id=f"S{len(sources) + 1}",
                title=(result.get("title") or "Untitled article").strip(),
                url=url.strip(),
                publisher=publisher,
                published_at=result.get("published_date") or result.get("published_at"),
                content=content[:5000] if isinstance(content, str) else "",
                search_perspectives=[perspective] if perspective else [],
            )
        )
    return sources


def search_sources(topic: str, api_key: str, max_results: int = 6, days: int | None = None) -> list[Source]:
    plan = ResearchPlan(queries=[{"query": topic, "perspective": "recent reporting"}])
    return search_queries(plan, api_key, max_results, days)


def _title_key(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def _mark_possible_duplicate_reporting(sources: list[Source]) -> list[Source]:
    groups: dict[int, str] = {}
    next_group = 1
    for left_index, left in enumerate(sources):
        left_title = _title_key(left.title)
        if len(left_title) < 12:
            continue
        for right_index in range(left_index + 1, len(sources)):
            right_title = _title_key(sources[right_index].title)
            if len(right_title) >= 12 and SequenceMatcher(None, left_title, right_title).ratio() >= 0.88:
                group_id = groups.get(left_index) or groups.get(right_index) or f"D{next_group}"
                if group_id == f"D{next_group}":
                    next_group += 1
                groups[left_index] = group_id
                groups[right_index] = group_id
    return [
        source.model_copy(update={"possible_duplicate_group": groups.get(index, source.possible_duplicate_group)})
        for index, source in enumerate(sources)
    ]


def search_queries(
    plan: ResearchPlan,
    api_key: str,
    max_results: int = 6,
    days: int | None = None,
    existing_sources: list[Source] | None = None,
) -> list[Source]:
    if not api_key:
        raise RetrievalError("TAVILY_API_KEY is not configured.")
    if not plan.queries:
        raise RetrievalError("Research plan contains no search queries.")

    try:
        client = TavilyClient(api_key=api_key)
    except Exception as exc:
        raise RetrievalError(f"Tavily client initialization failed: {exc}") from exc
    unique_sources = {
        _canonical_url(str(source.url)): source for source in (existing_sources or [])
    }
    errors = []
    per_query_limit = max(2, min(5, (max_results + len(plan.queries) - 1) // len(plan.queries)))
    for search_query in plan.queries:
        options = {"query": search_query.query, "topic": "news", "max_results": per_query_limit}
        if days:
            options["days"] = days
        try:
            response = client.search(**options)
        except Exception as exc:
            errors.append(f"{search_query.perspective.value}: {exc}")
            continue
        results = response.get("results", []) if isinstance(response, dict) else []
        for source in normalize_results(results if isinstance(results, list) else [], search_query.perspective.value):
            key = _canonical_url(str(source.url))
            existing = unique_sources.get(key)
            if existing:
                perspectives = sorted(set(existing.search_perspectives + source.search_perspectives))
                unique_sources[key] = existing.model_copy(update={"search_perspectives": perspectives})
            else:
                source = source.model_copy(update={"source_id": f"S{len(unique_sources) + 1}"})
                unique_sources[key] = source

    sources = list(unique_sources.values())[:max_results]
    if not sources and errors:
        raise RetrievalError("All Tavily searches failed: " + " | ".join(errors))
    return _mark_possible_duplicate_reporting(sources)