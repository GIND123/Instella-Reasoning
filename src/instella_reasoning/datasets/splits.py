"""Verified seen/unseen splits — the study's treatment assignment.

The contamination question that this repository originally asked ("are GSM8K *test*
items in Instella's training data?") is unanswerable by construction: the indexed corpus
is `amd/Instella-GSM8K-synthetic`, which is derived from GSM8K **train**. A scan of test
items against it returns zero contaminated items no matter how the thresholds are set,
so the headline contaminated-vs-clean contrast has an empty treatment group.

This module asks the answerable version instead:

    Does the model do better on problems it **provably memorised**?

GSM8K **train** items are verbatim in Instella's stage-2 data; GSM8K **test** items are
not. Both splits are same-distribution, same-annotator, same-difficulty, which makes the
pair a natural experiment with a treatment assignment that can be *verified* rather than
inferred from a cosine proxy.

Verification is exact, not semantic. For every candidate item we measure **n-gram
containment**: the fraction of the item's word 13-grams that occur somewhere in the
corpus. A memorised item scores near 1.0; an unseen item scores near 0.0. Items in the
middle are *excluded from both arms* rather than being forced into one — an ambiguous
treatment label is worse than a smaller sample.

Why 13-grams: it is the standard contamination window (Brown et al., 2020) and it is
already what :mod:`instella_reasoning.contamination` uses, so the two stages agree.

Scale note. The corpus is ~1.4M documents. Containment is computed in a single streaming
pass using a rolling polynomial hash, so cost is O(corpus tokens) with a small constant
and memory is O(item grams) — it does not materialise the corpus.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from instella_reasoning.records import BenchmarkItem, CorpusDocument

NGRAM_N = 13
#: Containment at or above this => the item is verifiably in the training corpus.
SEEN_THRESHOLD = 0.80
#: Containment at or below this => the item is verifiably absent.
UNSEEN_THRESHOLD = 0.10

_WORD = re.compile(r"[a-z0-9]+")
# Rolling hash over a 61-bit Mersenne prime: collisions are ~2^-61 per comparison, which
# is far below the noise floor of any conclusion drawn from these labels.
_MOD = (1 << 61) - 1
_BASE = 1_000_003


def normalize_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — punctuation- and whitespace-insensitive."""
    return _WORD.findall(text.lower())


def _token_ids(tokens: Iterable[str], vocab: dict[str, int]) -> list[int]:
    out = []
    for tok in tokens:
        tid = vocab.get(tok)
        if tid is None:
            tid = len(vocab) + 1
            vocab[tok] = tid
        out.append(tid)
    return out


def _gram_hashes(token_ids: list[int], n: int = NGRAM_N) -> Iterator[int]:
    """Rolling polynomial hashes of every n-gram, O(1) per window."""
    if len(token_ids) < n:
        return
    high = pow(_BASE, n - 1, _MOD)
    h = 0
    for tid in token_ids[:n]:
        h = (h * _BASE + tid) % _MOD
    yield h
    for i in range(n, len(token_ids)):
        h = ((h - token_ids[i - n] * high) * _BASE + token_ids[i]) % _MOD
        yield h


@dataclass(slots=True)
class ContainmentResult:
    """Exact-match evidence that one benchmark item is (or is not) in the corpus."""

    benchmark_id: str
    n_grams: int
    n_matched: int
    containment: float
    verdict: str  # "seen" | "unseen" | "ambiguous" | "too_short"
    example_source: str | None = None

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "n_grams": self.n_grams,
            "n_matched": self.n_matched,
            "containment": round(self.containment, 6),
            "verdict": self.verdict,
            "example_source": self.example_source,
        }


def verify_containment(
    items: list[BenchmarkItem],
    corpus: Iterable[CorpusDocument],
    n: int = NGRAM_N,
    seen_threshold: float = SEEN_THRESHOLD,
    unseen_threshold: float = UNSEEN_THRESHOLD,
    progress_every: int = 200_000,
) -> dict[str, ContainmentResult]:
    """Measure exact n-gram containment of every item against a streamed corpus.

    One pass over the corpus. ``corpus`` may be a generator, so a 1.4M-document JSONL is
    never held in memory.
    """
    vocab: dict[str, int] = {}
    # gram hash -> the items that contain it (one gram can belong to several items)
    gram_owners: dict[int, set[str]] = defaultdict(set)
    item_grams: dict[str, set[int]] = {}

    for item in items:
        ids = _token_ids(normalize_tokens(item.prompt), vocab)
        grams = set(_gram_hashes(ids, n))
        item_grams[item.id] = grams
        for g in grams:
            gram_owners[g].add(item.id)

    matched: dict[str, set[int]] = {item.id: set() for item in items}
    example: dict[str, str] = {}

    for doc_index, doc in enumerate(corpus):
        if progress_every and doc_index and doc_index % progress_every == 0:
            hits = sum(1 for k, v in matched.items() if v)
            print(f"[splits] scanned {doc_index:,} docs — {hits}/{len(items)} items with >=1 hit")
        ids = _token_ids(normalize_tokens(doc.text), vocab)
        for g in _gram_hashes(ids, n):
            owners = gram_owners.get(g)
            if not owners:
                continue
            for owner in owners:
                matched[owner].add(g)
                example.setdefault(owner, doc.source or doc.id)

    results: dict[str, ContainmentResult] = {}
    for item in items:
        total = len(item_grams[item.id])
        hit = len(matched[item.id])
        if total == 0:
            results[item.id] = ContainmentResult(item.id, 0, 0, 0.0, "too_short")
            continue
        containment = hit / total
        if containment >= seen_threshold:
            verdict = "seen"
        elif containment <= unseen_threshold:
            verdict = "unseen"
        else:
            verdict = "ambiguous"
        results[item.id] = ContainmentResult(
            item.id, total, hit, containment, verdict, example.get(item.id)
        )
    return results


@dataclass(slots=True)
class SeenUnseenSplit:
    """A difficulty-matched pair of verified arms, ready for generation."""

    seen: list[BenchmarkItem] = field(default_factory=list)
    unseen: list[BenchmarkItem] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)
    containment: dict[str, ContainmentResult] = field(default_factory=dict)

    def all_items(self) -> list[BenchmarkItem]:
        return [*self.seen, *self.unseen]

    @property
    def usable(self) -> bool:
        return bool(self.seen) and bool(self.unseen)

    def diagnosis(self) -> str | None:
        """Why the split came out empty, in the words of the thing to fix."""
        if self.usable:
            return None
        pool_seen = self.excluded.get("_pool_seen", 0)
        pool_unseen = self.excluded.get("_pool_unseen", 0)
        if pool_seen == 0 and pool_unseen == 0:
            return (
                "Neither arm has eligible items: no train item cleared the seen threshold "
                "and no test item cleared the unseen threshold. Check that the corpus is "
                "the one the model actually trained on."
            )
        if pool_seen == 0:
            return (
                "No train item was verified as SEEN. The corpus does not appear to contain "
                "the benchmark's train split — verify the corpus source before continuing."
            )
        if pool_unseen == 0:
            return (
                "No test item was verified as UNSEEN: every candidate landed in the "
                "ambiguous band. Test items that partially overlap the corpus usually mean "
                "the corpus contains near-duplicates of them, or the items are too "
                "templated to distinguish. Inspect the containment distribution (F4)."
            )
        return (
            f"Pools are non-empty (seen={pool_seen}, unseen={pool_unseen}) but no matched "
            "pair survived difficulty binning — try fewer bins."
        )

    def summary(self) -> dict:
        return {
            "n_seen": len(self.seen),
            "n_unseen": len(self.unseen),
            "usable": self.usable,
            "diagnosis": self.diagnosis(),
            "excluded": dict(sorted(self.excluded.items())),
            "containment_seen_mean": _mean(
                [self.containment[i.id].containment for i in self.seen if i.id in self.containment]
            ),
            "containment_unseen_mean": _mean(
                [
                    self.containment[i.id].containment
                    for i in self.unseen
                    if i.id in self.containment
                ]
            ),
        }


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def build_seen_unseen_split(
    train_items: list[BenchmarkItem],
    test_items: list[BenchmarkItem],
    containment: dict[str, ContainmentResult],
    n_per_arm: int,
    difficulty_bins: dict[str, int] | None = None,
    seed: int = 6198,
) -> SeenUnseenSplit:
    """Select ``n_per_arm`` verified-seen and verified-unseen items, difficulty-matched.

    Only train items whose containment clears ``seen`` and test items whose containment
    clears ``unseen`` are eligible — an item whose treatment status cannot be verified is
    dropped, and the reason is counted in ``excluded``.

    Matching is on the model-independent difficulty bin (GSM8K reasoning-step count) so
    the two arms have the same difficulty profile. Without this the seen/unseen contrast
    would be confounded by whatever difficulty difference the samples happened to have.
    """
    import random

    rng = random.Random(seed)
    excluded: dict[str, int] = defaultdict(int)

    def eligible(items: list[BenchmarkItem], want: str) -> list[BenchmarkItem]:
        out = []
        for item in items:
            res = containment.get(item.id)
            if res is None:
                excluded[f"{want}:unverified"] += 1
                continue
            if res.verdict != want:
                excluded[f"{want}:{res.verdict}"] += 1
                continue
            out.append(item)
        return out

    seen_pool = eligible(train_items, "seen")
    unseen_pool = eligible(test_items, "unseen")
    # Pool sizes are recorded even when they are zero. Matching takes the per-bin minimum
    # of the two pools, so one empty pool silently yields two empty arms — a failure that
    # is invisible unless the counts are reported.
    excluded["_pool_seen"] = len(seen_pool)
    excluded["_pool_unseen"] = len(unseen_pool)

    bins = difficulty_bins or {}

    def by_bin(pool: list[BenchmarkItem]) -> dict[int, list[BenchmarkItem]]:
        grouped: dict[int, list[BenchmarkItem]] = defaultdict(list)
        for item in pool:
            grouped[bins.get(item.id, 0)].append(item)
        for group in grouped.values():
            rng.shuffle(group)
        return grouped

    seen_bins, unseen_bins = by_bin(seen_pool), by_bin(unseen_pool)
    all_bins = sorted(set(seen_bins) | set(unseen_bins))

    # Take the same number from each difficulty bin in both arms, so the arms are matched
    # rather than merely equal-sized. Per-bin capacity is the smaller of the two pools.
    seen_out: list[BenchmarkItem] = []
    unseen_out: list[BenchmarkItem] = []
    if all_bins:
        target_per_bin = max(1, n_per_arm // len(all_bins))
        for bin_id in all_bins:
            take = min(target_per_bin, len(seen_bins.get(bin_id, [])), len(unseen_bins.get(bin_id, [])))
            seen_out.extend(seen_bins.get(bin_id, [])[:take])
            unseen_out.extend(unseen_bins.get(bin_id, [])[:take])
        # Top up from whatever remains, still pairwise, until the target is reached.
        leftovers_seen = [i for b in all_bins for i in seen_bins.get(b, [])[target_per_bin:]]
        leftovers_unseen = [i for b in all_bins for i in unseen_bins.get(b, [])[target_per_bin:]]
        extra = min(n_per_arm - len(seen_out), len(leftovers_seen), len(leftovers_unseen))
        if extra > 0:
            seen_out.extend(leftovers_seen[:extra])
            unseen_out.extend(leftovers_unseen[:extra])

    for item in seen_out:
        item.metadata = {**item.metadata, "arm": "seen", "gsm8k_split": "train"}
    for item in unseen_out:
        item.metadata = {**item.metadata, "arm": "unseen", "gsm8k_split": "test"}

    return SeenUnseenSplit(
        seen=seen_out[:n_per_arm],
        unseen=unseen_out[:n_per_arm],
        excluded=dict(excluded),
        containment=containment,
    )


def balance_report(split: SeenUnseenSplit, difficulty_bins: dict[str, int]) -> dict:
    """Difficulty balance across the two arms — the evidence the match actually worked.

    A reviewer will ask whether the seen arm is simply easier. This is the table that
    answers it, and the suite prints it before spending any GPU time.
    """
    def profile(items: list[BenchmarkItem]) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for item in items:
            counts[difficulty_bins.get(item.id, 0)] += 1
        return dict(sorted(counts.items()))

    seen_profile = profile(split.seen)
    unseen_profile = profile(split.unseen)
    max_gap = 0
    for bin_id in set(seen_profile) | set(unseen_profile):
        max_gap = max(max_gap, abs(seen_profile.get(bin_id, 0) - unseen_profile.get(bin_id, 0)))
    # A gap of 1 is arithmetic, not confounding: an odd pool size cannot split evenly
    # across bins. Flagging it would train the operator to ignore the warning, which is
    # worse than not having one.
    return {
        "seen_difficulty_profile": seen_profile,
        "unseen_difficulty_profile": unseen_profile,
        "max_per_bin_imbalance": max_gap,
        "balanced": max_gap <= 1,
    }
