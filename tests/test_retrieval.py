import retrieval
from models import ResearchPlan
from retrieval import normalize_results, search_queries


def test_normalize_results_removes_duplicate_urls_and_assigns_ids():
    results = [
        {"url": "https://example.com/story?utm_source=feed", "title": "First", "content": "Text"},
        {"url": "https://example.com/story/", "title": "Duplicate", "content": "Other text"},
        {"url": "javascript:alert(1)", "title": "Invalid", "content": "Ignore"},
    ]

    sources = normalize_results(results)

    assert len(sources) == 1
    assert sources[0].source_id == "S1"
    assert sources[0].publisher == "example.com"


def test_normalize_results_handles_empty_input():
    assert normalize_results([]) == []


def test_search_queries_merges_urls_and_tracks_query_perspectives(monkeypatch):
    class FakeTavilyClient:
        def __init__(self, api_key):
            self.queries = []

        def search(self, query, **kwargs):
            self.queries.append(query)
            if query == "background query":
                return {
                    "results": [
                        {
                            "url": "https://example.com/story?utm_source=one",
                            "title": "The New Climate Policy Explained",
                            "content": "Background reporting.",
                        },
                        {
                            "url": "https://publisher.test/story",
                            "title": "The New Climate Policy Explained",
                            "content": "Related coverage.",
                        },
                    ]
                }
            return {
                "results": [
                    {
                        "url": "https://example.com/story/",
                        "title": "The New Climate Policy Explained",
                        "content": "A second perspective.",
                    }
                ]
            }

    monkeypatch.setattr(retrieval, "TavilyClient", FakeTavilyClient)
    plan = ResearchPlan(
        queries=[
            {"query": "background query", "perspective": "background"},
            {"query": "opposing query", "perspective": "opposing viewpoints"},
        ]
    )

    sources = search_queries(plan, api_key="test", max_results=6)

    assert len(sources) == 2
    assert sources[0].search_perspectives == ["background", "opposing viewpoints"]
    assert sources[0].possible_duplicate_group == sources[1].possible_duplicate_group