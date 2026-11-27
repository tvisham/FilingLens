"""The free local classifier (no LLM): categories, extractive summaries, citations."""

from __future__ import annotations

from filinglens.risk.classify import Category, FactorInput, validate_batch
from filinglens.risk.embed import HashingEmbedder
from filinglens.risk.local_classify import LOCAL_MODEL, LocalClassifier, extractive_summary


def test_extractive_summary_prefers_a_sentence_heading() -> None:
    heading = "The Company depends on suppliers in Asia for most components."
    assert extractive_summary(heading, heading + "\n\nMore text.") == heading
    # Title-style heading: fall back to the body's first sentence.
    assert extractive_summary("Supply Chain", "Supply Chain\n\nWe rely on one supplier. It is in Asia.") == (
        "We rely on one supplier."
    )
    long = "word " * 200
    assert len(extractive_summary(None, long)) <= 300


def test_classify_categories_and_citations() -> None:
    clf = LocalClassifier(HashingEmbedder())
    factors = [
        FactorInput(
            "F1",
            "Cybersecurity attacks and data breaches could harm us.",
            "Hackers, ransomware and unauthorized access to systems and confidential information.",
            (7, 8),
        ),
        FactorInput(
            "F2",
            "Climate change and the energy transition may raise costs.",
            "Climate change, energy transition, environmental regulation and emissions rules.",
            (9,),
        ),
        FactorInput("F3", None, "no chunks", ()),
    ]
    run = clf.classify(factors)
    assert run.results["F1"].categories[0] is Category.CYBER
    assert run.results["F2"].categories[0] is Category.ESG_CLIMATE
    assert run.results["F1"].chunk_ids == [7] and run.results["F1"].severity is None
    assert "F3" in run.failures  # nothing to cite: never stored as classified
    assert run.served_by == {LOCAL_MODEL} and clf.spent_usd == 0


def test_local_results_satisfy_the_llm_output_rules() -> None:
    """Same validation as the LLM path, apart from severity (which it leaves empty)."""
    clf = LocalClassifier(HashingEmbedder())
    f = FactorInput(
        "F1", "Competitors may cut prices and take market share.", "Competitors and pricing.", (3, 4)
    )
    c = clf.classify([f]).results["F1"]
    raw = {
        "results": [
            {
                "key": "F1",
                "categories": [x.value for x in c.categories],
                "severity": 3,
                "summary": c.summary,
                "chunk_ids": c.chunk_ids,
            }
        ]
    }
    _, failed = validate_batch(raw, [f])
    assert failed == {} and 1 <= len(c.categories) <= 3


def test_summary_never_a_punctuation_fragment() -> None:
    # Run-in lead "Economic conditions" whose bold run stopped before its period.
    s = extractive_summary(
        "Economic conditions", "Economic conditions\n\n. Demand for energy tracks the economy."
    )
    assert s == "Demand for energy tracks the economy."
    assert extractive_summary("Security concerns", "Security concerns\n\n.") == "Security concerns"
