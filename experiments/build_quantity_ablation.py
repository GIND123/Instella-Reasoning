#!/usr/bin/env python
"""Build the quantity-ablation suite: is it the *number* that breaks the model, or the text?

Why this exists
---------------
The earlier finding was that inserting a sentence carrying an unused number costs 6-8
accuracy points, while inserting a plain distractor sentence costs nothing. Measured on
the `atlas-v1` arms that comparison is:

    irrelevant_context   "The weather that day was pleasant."                    5 words, off-topic, no number
    distractor_quantity  "An unrelated shipment of 47 boxes arrived on the       15 words, domain-ish, number
                          same day and was never opened."

Three things differ, not one: sentence length, topical relatedness to the problem, and
the presence of a number. Any referee reads that table and asks which of the three did
the work. So the effect is real but the *attribution* is not identified.

This module replaces that comparison with a crossed design in which the four inserted
sentences are the same sentence. One frame, twenty words, two slots. Note the frame puts the noun in a prepositional
phrase so that singular and plural nouns are both grammatical in it:

    "A separate log listing {qty} entries about {topic} was filed in another office
     and has no bearing on this question."

      condition          {topic}                         {qty}
      offtopic_noqty     an off-topic noun               "several"
      domain_noqty       a noun lifted from the problem  "several"
      offtopic_qty       an off-topic noun               an integer
      domain_qty         a noun lifted from the problem  an integer

Word count is identical in all four. Syntax is identical. Insertion position is
identical. Topicality and number-presence are crossed rather than confounded, so the
main effect of each is estimable and so is the interaction. That is the whole point.

Two further conditions:

* ``orig`` - the untouched problem, the paired baseline for everything else.
* ``reorder_safe`` - premise reordering restricted to items where reordering cannot
  change the problem. The existing ``premise_reordering`` generator shuffles premises
  unconditionally, which for narrative word problems destroys anaphora and temporal
  reference: "This week she sent 50 less than double what she sent last week" placed
  before the sentence introducing last week is not a reordered problem, it is a broken
  one, and a human would fail it too. Measured on the atlas arms, the large reordering
  effect largely disappears once items with cross-sentence dependencies are excluded.
  This condition tests order-sensitivity on items where order genuinely does not matter.

Gold is preserved by construction in every condition, and ``--validate`` re-checks it
rather than trusting the construction.

Usage
-----
    python experiments/build_quantity_ablation.py \\
        --parents experiments/runs/ckpt-axis-v1/base/gsm8k_train_parents.jsonl \\
                  experiments/runs/ckpt-axis-v1/base/gsm8k_test_parents.jsonl \\
        --out-dir experiments/runs/qty-ablation-v1/base \\
        --n-parents 1000 --seed 6198
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------------
# The frame. Every inserted sentence is this string with two slots filled, so the four
# insertion conditions differ by exactly two words and nothing else.
# ---------------------------------------------------------------------------------
FRAME = (
    "A separate log listing {qty} entries about {topic} was filed in another office "
    "and has no bearing on this question."
)
NO_QUANTITY_FILLER = "several"

#: Nouns that cannot plausibly belong to a grade-school word problem's domain. Chosen to
#: be concrete and countable so the frame stays grammatical with each of them.
OFFTOPIC_NOUNS = (
    "weather", "glacier", "seabed", "pollen", "tectonic", "auroral",
    "sunspot", "meteorite", "volcanic", "planetary",
)

#: Tokens that carry no domain information, so they must never be picked as the
#: problem's representative noun.
_STOPWORDS = frozenset("""
about above after again against all also am an and any are as at be because been before
being below between both but by can cannot could did do does doing down during each few
for from further had has have having he her here hers herself him himself his how if in
into is it its itself just me more most my myself no nor not now of off on once only or
other ought our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why with would you your yours
yourself yourselves many much total number does get gets got make makes made take takes
day days week weeks month months year years time times left over each every another
""".split())

_PRONOUN = re.compile(r"\b(she|he|it|they|them|his|her|hers|their|theirs|its)\b", re.I)
_TEMPORAL = re.compile(
    r"\b(then|later|after|afterwards?|next|following|subsequently|finally|now|"
    r"first|second|third|earlier)\b",
    re.I,
)
_BACKREF = re.compile(
    r"\b(half as|twice as|three times as|as many|as much|the same|that many|the rest|"
    r"the remaining|more than|fewer than|less than|each of them|of these|of those|"
    r"this|these|those|remaining|leftover|"
    # Bare determiners are safe where the author put them and unsafe once permuted:
    # "Each bag was $6.00" reads fine after the bags are introduced and not before.
    r"each|every|both|another|the other|the first|the second|the third)\b",
    re.I,
)
_QUESTION = re.compile(r"\?\s*$|^(how|what|which|who|when|where|why|calculate|find|"
                       r"determine|compute)\b", re.I)

CONDITIONS = (
    "orig",
    "offtopic_noqty",
    "domain_noqty",
    "offtopic_qty",
    "domain_qty",
    "reorder_safe",
)


# ---------------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------------
#: Abbreviations whose trailing period must not end a sentence. Without this, "Mrs.
#: Tatiana owns a grocery store" splits into "Mrs." and "Tatiana owns ...", and the
#: reordering condition then separates the title from the name and produces gibberish
#: that scores as a reasoning failure. Found by the validator, not by inspection.
_ABBREV = (
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "mt", "no", "vs",
    "approx", "est", "fig", "inc", "ltd", "co", "etc", "e.g", "i.e", "oz", "lb", "lbs",
)
_ABBREV_RE = re.compile(
    r"(?:\b(?:" + "|".join(re.escape(a) for a in _ABBREV) + r")\.)$|"
    r"(?:\b[A-Z]\.)$|"          # single-letter initials: "J. Smith"
    r"(?:\d\.)$",               # decimals split by a following capital
    re.I,
)


def split_sentences(text: str) -> list[str]:
    """Sentence split that does not break on honorifics, initials, or decimals."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    merged: list[str] = []
    for part in raw:
        if not part.strip():
            continue
        if merged and _ABBREV_RE.search(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def is_question(sentence: str) -> bool:
    return bool(_QUESTION.search(sentence.strip()) or _QUESTION.match(sentence.strip()))


def question_index(sentences: list[str]) -> int:
    """Index of the interrogative, searching from the end. Falls back to the last."""
    for i in range(len(sentences) - 1, -1, -1):
        if is_question(sentences[i]):
            return i
    return len(sentences) - 1


def prompt_integers(text: str) -> set[int]:
    out: set[int] = set()
    for tok in re.findall(r"\d[\d,]*", text):
        try:
            out.add(int(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def gold_int(answer: object) -> int | None:
    if answer is None:
        return None
    m = re.search(r"-?\d[\d,]*", str(answer))
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def domain_noun(prompt: str) -> str | None:
    """The problem's most representative content noun.

    Frequency over lowercase alphabetic tokens of length >= 4 that are not stopwords,
    tie-broken by first occurrence so the choice is deterministic. Capitalised tokens are
    skipped because they are usually the actor's name, which would make the inserted
    sentence read as though it were about the actor and therefore *relevant*.
    """
    tokens = re.findall(r"\b[a-z]{4,}\b", prompt)
    counts = Counter(t for t in tokens if t not in _STOPWORDS)
    if not counts:
        return None
    # Prefer a plural noun: the frame reads "entries about {topic}", and an adjective or
    # verb in that slot ("entries about painted") is not English. Plurals are the safest
    # surface cue for a noun without a tagger. Fall back to raw frequency.
    plurals = {t: c for t, c in counts.items() if t.endswith("s") and not t.endswith("ss")}
    pool = plurals or counts
    best = max(pool.values())
    for tok in tokens:
        if pool.get(tok) == best:
            return tok
    return None


def pick_number(prompt: str, gold: int | None, rng: random.Random) -> int | None:
    """A two-digit integer colliding with nothing the problem already contains.

    Two digits keeps the inserted magnitude comparable across items, so the condition
    cannot be confounded by answer magnitude the way raw recall rates were.
    """
    taken = prompt_integers(prompt)
    if gold is not None:
        taken.add(abs(gold))
    candidates = [v for v in range(13, 97) if v not in taken]
    return rng.choice(candidates) if candidates else None


def insert_before_question(prompt: str, sentence: str) -> str | None:
    """Place the sentence among the premises, immediately before the interrogative."""
    sents = split_sentences(prompt)
    if len(sents) < 2:
        return None
    qi = question_index(sents)
    rebuilt = sents[:qi] + [sentence] + sents[qi:]
    return " ".join(s.strip() for s in rebuilt)


def order_safe(prompt: str) -> bool:
    """True when permuting the premises cannot change what the problem means.

    Conservative by design: any pronoun, temporal connective, or back-reference in a
    premise other than the first means a later sentence may depend on an earlier one, so
    the item is rejected. Rejecting a safe item costs sample size; accepting an unsafe
    one costs the validity of the whole condition.
    """
    sents = split_sentences(prompt)
    if len(sents) < 3:
        return False
    qi = question_index(sents)
    head = sents[:qi]                       # premises only; anything after the
    if len(head) < 2:                       # interrogative is a trailing instruction
        return False                        # ("(Round to the nearest integer)") and is
    for sentence in head:                   # never a premise to permute
        if len(sentence.split()) < 4 or not sentence[:1].isupper():
            return False                    # fragment: the splitter mis-fired here
    for sentence in head:
        if _PRONOUN.search(sentence) or _TEMPORAL.search(sentence) or _BACKREF.search(sentence):
            return False
    return True


def reorder_premises(prompt: str, rng: random.Random) -> str | None:
    """Permute the premises only. The interrogative and anything after it stay put."""
    sents = split_sentences(prompt)
    qi = question_index(sents)
    head, tail = sents[:qi], sents[qi:]
    body = head
    shuffled = body[:]
    for _ in range(8):
        rng.shuffle(shuffled)
        if shuffled != body:
            break
    else:
        return None
    return " ".join(s.strip() for s in shuffled + tail)


# ---------------------------------------------------------------------------------
# variant construction
# ---------------------------------------------------------------------------------
def build_variants(item: dict, rng_seed: int) -> tuple[list[dict], list[str]]:
    """Return (variants, skipped_condition_names) for one parent problem."""
    prompt = item["prompt"]
    gold = gold_int(item.get("answer"))
    variants: list[dict] = []
    skipped: list[str] = []

    def emit(cond: str, text: str, extra: dict) -> None:
        meta = dict(item.get("metadata") or {})
        meta.update({
            "condition": cond,
            "parent_answer": gold,
            "ablation": "quantity",
            "answer_changing": False,
        })
        meta.update(extra)
        variants.append({
            "id": f"{item['id']}__{cond}",
            "parent_id": item["id"],
            "prompt": text,
            "answer": item.get("answer"),
            "variant_type": cond,
            "metadata": meta,
        })

    emit("orig", prompt, {"inserted_sentence": None})

    noun = domain_noun(prompt)
    number = pick_number(prompt, gold, random.Random(rng_seed))
    if noun is None:
        skipped.append("domain_*: no usable content noun")
    if number is None:
        skipped.append("*_qty: no free two-digit integer")

    offtopic = random.Random(rng_seed + 1).choice(OFFTOPIC_NOUNS)

    cells = (
        ("offtopic_noqty", offtopic, NO_QUANTITY_FILLER, offtopic is not None),
        ("domain_noqty", noun, NO_QUANTITY_FILLER, noun is not None),
        ("offtopic_qty", offtopic, str(number), number is not None),
        ("domain_qty", noun, str(number), noun is not None and number is not None),
    )
    for cond, topic, qty, ok in cells:
        if not ok:
            skipped.append(cond)
            continue
        sentence = FRAME.format(qty=qty, topic=topic)
        text = insert_before_question(prompt, sentence)
        if text is None:
            skipped.append(f"{cond}: single-sentence prompt")
            continue
        emit(cond, text, {
            "inserted_sentence": sentence,
            "inserted_topic": topic,
            "inserted_topicality": "domain" if topic == noun else "offtopic",
            "inserted_number": number if qty != NO_QUANTITY_FILLER else None,
            "has_number": qty != NO_QUANTITY_FILLER,
        })

    if order_safe(prompt):
        text = reorder_premises(prompt, random.Random(rng_seed + 2))
        if text is not None:
            emit("reorder_safe", text, {
                "inserted_sentence": None,
                "order_safe_heuristic": True,
                "exploratory": True,
                "requires_manual_audit": True,
            })
        else:
            skipped.append("reorder_safe: permutation collapsed to identity")
    else:
        skipped.append("reorder_safe: cross-sentence dependency")

    return variants, skipped


# ---------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------
def validate(parents: dict[str, dict], variants: list[dict]) -> dict:
    """Re-derive every invariant the construction is supposed to guarantee."""
    problems: list[str] = []
    by_parent: dict[str, dict[str, dict]] = {}
    for v in variants:
        by_parent.setdefault(v["parent_id"], {})[v["metadata"]["condition"]] = v

    word_counts: Counter = Counter()
    n_checked = 0
    for pid, conds in by_parent.items():
        parent = parents[pid]
        for cond, v in conds.items():
            n_checked += 1
            if str(v["answer"]) != str(parent.get("answer")):
                problems.append(f"{v['id']}: gold changed")
            if cond == "orig":
                if v["prompt"] != parent["prompt"]:
                    problems.append(f"{v['id']}: orig prompt altered")
                continue
            if cond == "reorder_safe":
                if sorted(split_sentences(v["prompt"])) != sorted(split_sentences(parent["prompt"])):
                    problems.append(f"{v['id']}: reorder changed the sentence set")
                continue
            sent = v["metadata"]["inserted_sentence"]
            if sent not in v["prompt"]:
                problems.append(f"{v['id']}: inserted sentence missing")
            word_counts[len(sent.split())] += 1
            n = v["metadata"].get("inserted_number")
            if n is not None and n in prompt_integers(parent["prompt"]):
                problems.append(f"{v['id']}: inserted number {n} collides with the problem")
            if n is None and re.search(r"\d", sent):
                problems.append(f"{v['id']}: no-quantity cell carries a digit")

    # The 2x2 is only a 2x2 if all four cells are present for the same parent.
    complete = sum(
        1 for conds in by_parent.values()
        if all(c in conds for c in ("offtopic_noqty", "domain_noqty", "offtopic_qty", "domain_qty"))
    )
    return {
        "n_parents": len(by_parent),
        "n_variants": len(variants),
        "n_checked": n_checked,
        "n_parents_with_complete_2x2": complete,
        "inserted_sentence_word_counts": dict(word_counts),
        "n_problems": len(problems),
        "problems": problems[:20],
    }


# ---------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-parents", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--tag", default="qty_ablation")
    args = ap.parse_args()

    parents: list[dict] = []
    for path in args.parents:
        split = "train" if "train" in Path(path).name else "test"
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["id"] = f"{split}__{row['id']}"      # namespaced: train and test reuse ids
            row.setdefault("metadata", {})["split"] = split
            parents.append(row)

    eligible = [p for p in parents if gold_int(p.get("answer")) is not None
                and len(split_sentences(p["prompt"])) >= 2]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    chosen = eligible[:args.n_parents]

    variants: list[dict] = []
    skip_counter: Counter = Counter()
    for i, item in enumerate(chosen):
        vs, skipped = build_variants(item, args.seed + i)
        variants.extend(vs)
        for s in skipped:
            skip_counter[s.split(":")[0]] += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / f"{args.tag}.jsonl"
    with open(items_path, "w", encoding="utf-8") as fh:
        for v in variants:
            fh.write(json.dumps(v) + "\n")

    report = validate({p["id"]: p for p in chosen}, variants)
    report.update({
        "seed": args.seed,
        "n_parents_pool": len(parents),
        "n_parents_eligible": len(eligible),
        "n_parents_requested": args.n_parents,
        "skipped_by_condition": dict(skip_counter),
        "items_path": str(items_path),
        "conditions": list(CONDITIONS),
        "frame": FRAME,
    })
    (out_dir / f"{args.tag}_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"[qty-ablation] {report['n_parents']} parents -> {report['n_variants']} variants")
    print(f"[qty-ablation] complete 2x2 on {report['n_parents_with_complete_2x2']} parents")
    print(f"[qty-ablation] inserted-sentence word counts: {report['inserted_sentence_word_counts']}")
    print(f"[qty-ablation] skipped: {dict(skip_counter)}")
    print(f"[qty-ablation] validation problems: {report['n_problems']}")
    for p in report["problems"]:
        print(f"    ! {p}")
    print(f"[qty-ablation] wrote {items_path}")
    return 1 if report["n_problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
