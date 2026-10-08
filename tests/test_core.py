from core import (
    Passage,
    PythonDocsIndex,
    expand_spanish_query,
    make_evidence,
    validate_citations,
)


def test_spanish_query_expansion_adds_python_terms():
    query = expand_spanish_query("¿Cómo recorro una lista con un bucle?")
    assert "for while" in query
    assert "list data structures" in query


def test_index_retrieves_matching_official_passage():
    passages = [
        Passage("For statements", "https://docs.python.org/3/tutorial/controlflow.html", "The for statement iterates over the items of any sequence."),
        Passage("Exceptions", "https://docs.python.org/3/tutorial/errors.html", "The try statement works with exceptions and error handling."),
    ]
    index = PythonDocsIndex(passages)
    results = index.search("for statement iterates sequence", top_k=2)
    assert results[0][0].title == "For statements"
    assert results[0][1] > 0


def test_evidence_and_unknown_citation_validation():
    passage = Passage("For statements", "https://docs.python.org/3/tutorial/controlflow.html", "for iterates over items")
    evidence, refs = make_evidence([(passage, 0.5)])
    assert "[D1]" in evidence
    assert refs["D1"].url == passage.url
    assert validate_citations("Está documentado [D1], pero [D9] no.", {"D1"}) == (
        "Está documentado [D1], pero [referencia no verificada] no."
    )
