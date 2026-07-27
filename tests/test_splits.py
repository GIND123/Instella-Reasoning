"""Verified seen/unseen split construction — the study's treatment assignment.

If these labels are wrong the headline contrast is meaningless, so the tests check the
properties the design depends on: exact containment separates planted from unplanted
items, ambiguous items are *excluded* rather than forced into an arm, and the two arms
come out difficulty-matched.
"""

from __future__ import annotations

from instella_reasoning.datasets.splits import (
    balance_report,
    build_seen_unseen_split,
    verify_containment,
)
from instella_reasoning.difficulty import assign_difficulty_bins
from instella_reasoning.records import BenchmarkItem, CorpusDocument

# Distinct vocabulary per item. Containment is measured on 13-grams, so fixture items that
# differ only in a couple of numbers would share almost all their n-grams and every item
# would look "seen" once any one of them is planted — a fixture artefact that would hide
# real regressions. Real GSM8K items differ in nearly every content word, like these.
_SUBJECTS = [
    ("Mira", "beekeeper", "hives", "jars of clover honey", "the Thursday market"),
    ("Otto", "luthier", "workshops", "spruce soundboards", "the violin fair"),
    ("Priya", "cartographer", "studios", "linen survey charts", "the harbour office"),
    ("Sven", "glassblower", "furnaces", "cobalt drinking goblets", "the winter guild"),
    ("Nadia", "falconer", "mews", "braided leather jesses", "the highland auction"),
    ("Tomas", "chandler", "cellars", "beeswax pillar candles", "the abbey kitchen"),
    ("Ines", "wheelwright", "yards", "elm cart hubs", "the drovers' road"),
    ("Kofi", "dyer", "vats", "indigo woollen skeins", "the cloth hall"),
    ("Lena", "clockmaker", "benches", "brass escapement wheels", "the observatory"),
    ("Rafael", "vintner", "presses", "oak-aged barrels", "the autumn tasting"),
    ("Yuki", "papermaker", "sheds", "mulberry writing sheets", "the scriptorium"),
    ("Abel", "farrier", "forges", "iron carriage shoes", "the coaching inn"),
]


def _items(prefix: str, n: int, steps: int = 2) -> list[BenchmarkItem]:
    out = []
    # Offset the subject by the prefix so the "train" and "test" pools never draw the same
    # scenario. Two items that share everything but one word overlap on almost every
    # 13-gram, which would make a planted train item drag its test twin above threshold.
    offset = sum(ord(c) for c in prefix)
    for i in range(n):
        name, trade, place, goods, venue = _SUBJECTS[(i + offset) % len(_SUBJECTS)]
        prompt = (
            f"{name} the {trade} of {prefix} keeps {i + 2} {place} and produced "
            f"{i + 7} {goods} last season, then delivered {i + 3} of them to {venue} "
            f"before the {prefix} inspectors arrived to record the {trade}'s tally."
        )
        rationale = "".join(f"<<{j + 2}+{j + 1}={2 * j + 3}>>" for j in range(steps))
        out.append(
            BenchmarkItem(
                id=f"{prefix}_{i:03d}",
                prompt=prompt,
                answer="1",
                parent_id=f"{prefix}_{i:03d}",
                metadata={"benchmark": "gsm8k", "rationale": rationale},
            )
        )
    return out


def _noise(n: int) -> list[CorpusDocument]:
    return [
        CorpusDocument(
            id=f"noise_{i}",
            text="Unrelated encyclopedia text about tidal patterns and harbour maintenance.",
            source="noise",
        )
        for i in range(n)
    ]


def test_planted_items_verify_as_seen_and_others_as_unseen() -> None:
    train, test = _items("train", 10), _items("test", 10)
    corpus = [
        CorpusDocument(id=f"d{i}", text=f"Worked example. {item.prompt} The answer is 1.", source="synthetic")
        for i, item in enumerate(train)
    ] + _noise(50)

    results = verify_containment([*train, *test], corpus)
    assert all(results[i.id].verdict == "seen" for i in train)
    assert all(results[i.id].verdict == "unseen" for i in test)
    assert all(results[i.id].containment == 1.0 for i in train)


def test_partial_overlap_is_excluded_rather_than_forced_into_an_arm() -> None:
    """A treatment label that cannot be verified is worse than a smaller sample."""
    items = _items("amb", 3)
    # Plant only the first half of each prompt, so containment lands between the thresholds.
    corpus = [
        CorpusDocument(id=f"d{i}", text=" ".join(item.prompt.split()[: len(item.prompt.split()) // 2]), source="s")
        for i, item in enumerate(items)
    ] + _noise(20)
    results = verify_containment(items, corpus)
    verdicts = {results[i.id].verdict for i in items}
    assert verdicts <= {"ambiguous", "unseen"}

    split = build_seen_unseen_split(items, [], results, n_per_arm=3)
    assert split.seen == []
    assert any(key.startswith("seen:") for key in split.excluded)


def test_arms_are_difficulty_matched() -> None:
    train = _items("train", 12, steps=2) + _items("trainhard", 12, steps=5)
    test = _items("test", 12, steps=2) + _items("testhard", 12, steps=5)
    corpus = [
        CorpusDocument(id=f"d{i}", text=item.prompt, source="s") for i, item in enumerate(train)
    ] + _noise(30)

    results = verify_containment([*train, *test], corpus)
    bins = assign_difficulty_bins([*train, *test], n_bins=2)
    split = build_seen_unseen_split(train, test, results, n_per_arm=16, difficulty_bins=bins)

    assert len(split.seen) == len(split.unseen)
    report = balance_report(split, bins)
    assert report["balanced"], report


def test_arm_metadata_is_stamped_for_the_analysis() -> None:
    train, test = _items("train", 4), _items("test", 4)
    corpus = [CorpusDocument(id=f"d{i}", text=i2.prompt, source="s") for i, i2 in enumerate(train)]
    results = verify_containment([*train, *test], corpus + _noise(10))
    split = build_seen_unseen_split(train, test, results, n_per_arm=4)
    assert {i.metadata["arm"] for i in split.seen} == {"seen"}
    assert {i.metadata["arm"] for i in split.unseen} == {"unseen"}
    assert {i.metadata["gsm8k_split"] for i in split.seen} == {"train"}


def test_short_items_are_flagged_not_silently_counted() -> None:
    short = [BenchmarkItem(id="s0", prompt="Too short.", answer="1", parent_id="s0")]
    results = verify_containment(short, _noise(5))
    assert results["s0"].verdict == "too_short"
