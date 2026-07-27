"""The Instella-3B developmental trajectory — the study's model axis.

AMD publishes *every* checkpoint in the Instella-3B pipeline, not just the endpoints.
That turns what is usually an observational comparison ("a 1B model vs a 3B model") into
a sequence of **controlled data interventions**: same architecture, same tokenizer, same
codebase, one documented data change per step.

    Stage1 ──Stage-2 data──▶ Instella-3B ──SFT──▶ Instella-3B-SFT ──DPO──▶ Instella-3B-Instruct
                                                                                    │
                                                                              math SFT│
                                                                                    ▼
                                                        Instella-3B-Math ◀──RL── Instella-3B-Math-SFT

The load-bearing fact for this study: **Stage 2 is where GSM8K-targeted data enters**
(Dolmino-Mix-1124, Tulu-3-SFT-Mixture, and `amd/Instella-GSM8K-synthetic`, which is
derived from GSM8K *train*; see arXiv:2511.10628 §Stage 2). ``Instella-3B-Stage1`` is
therefore the only checkpoint that has provably **not** seen GSM8K-derived text, which
makes it the control arm for the memorisation contrast in
:mod:`instella_reasoning.datasets.splits`.

``saw_gsm8k_derived_data`` encodes that fact per checkpoint. Nothing downstream should
hard-code model names; select through :func:`resolve` / :func:`tier` so a run can be
re-scoped without editing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Base (non-instruction-tuned) checkpoints need few-shot prompting: zero-shot CoT on a
# base model produces degenerate loops (observed: 408/556 outputs on AMD-OLMo-1B).
# Long-CoT checkpoints need a larger token budget: the Math checkpoint emitted its
# final-answer marker in 2% of outputs at 512 tokens, which invalidated its accuracy.


@dataclass(slots=True)
class Checkpoint:
    """One node in the Instella training trajectory."""

    tag: str  # short handle used in run dirs and CLI flags
    hf_id: str  # Hugging Face model id (or a local path for assembled copies)
    step: int  # position in the trajectory, 0 = earliest
    intervention: str  # what changed relative to the previous step
    is_base: bool  # True => no chat template, needs few-shot prompting
    saw_gsm8k_derived_data: bool  # provably exposed to GSM8K-train-derived text
    max_new_tokens: int = 1024  # per-model generation budget
    tier: int = 1  # 1 = core run, 2 = added if budget allows, 3 = appendix only
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    #: Some Instella repos ship remote code written for vLLM (their modeling_instella.py
    #: imports vllm unconditionally), so transformers cannot load them directly. These are
    #: assembled into a local HF-loadable copy first — see
    #: ``experiments/fetch_instella_math_hf.py``. When set, generation uses this path.
    local_assembly: str | None = None

    @property
    def load_path(self) -> str:
        """What to hand to ``--model``: the assembled local copy when one is required."""
        return self.local_assembly or self.hf_id

    @property
    def n_shot(self) -> int:
        """Few-shot exemplars. Base checkpoints loop under zero-shot CoT; 4 fixes that."""
        return 4 if self.is_base else 0


INSTELLA_TRAJECTORY: tuple[Checkpoint, ...] = (
    Checkpoint(
        tag="stage1",
        hf_id="amd/Instella-3B-Stage1",
        step=0,
        intervention="pretraining stage 1 only (4.07T tokens, OLMoE-mix-0924)",
        is_base=True,
        saw_gsm8k_derived_data=False,
        tier=1,
        notes="THE CONTROL ARM: the only checkpoint with no GSM8K-targeted data.",
        aliases=("s1",),
    ),
    Checkpoint(
        tag="stage2",
        hf_id="amd/Instella-3B",
        step=1,
        intervention="+ stage-2 mix (Dolmino, Tulu-3, Instella-GSM8K-synthetic)",
        is_base=True,
        saw_gsm8k_derived_data=True,
        tier=1,
        notes="THE TREATED ARM: Stage 2 explicitly targets GSM8K.",
        aliases=("base", "instella-3b"),
    ),
    Checkpoint(
        tag="sft",
        hf_id="amd/Instella-3B-SFT",
        step=2,
        intervention="+ supervised fine-tuning",
        is_base=False,
        saw_gsm8k_derived_data=True,
        tier=2,
        notes="Separates SFT from DPO in the post-training effect.",
    ),
    Checkpoint(
        tag="instruct",
        hf_id="amd/Instella-3B-Instruct",
        step=3,
        intervention="+ DPO",
        is_base=False,
        saw_gsm8k_derived_data=True,
        tier=1,
        notes="Strongest general model; the headline accuracy number.",
    ),
    Checkpoint(
        tag="math_sft",
        hf_id="amd/Instella-3B-Math-SFT",
        step=4,
        intervention="+ math-focused SFT",
        is_base=False,
        saw_gsm8k_derived_data=True,
        max_new_tokens=1536,
        tier=2,
        notes="Separates math SFT from the RL step in the A6 story.",
        local_assembly="models/Instella-3B-Math-SFT-hf",
    ),
    Checkpoint(
        tag="math",
        hf_id="amd/Instella-3B-Math",
        step=5,
        intervention="+ RL",
        is_base=False,
        saw_gsm8k_derived_data=True,
        # Long-CoT self-verifying reasoner. At 512 tokens it emitted the '####' marker in
        # 2% of outputs and 26% of its responses contained the gold answer but scored
        # wrong. 1536 is a measurement-validity requirement, not a nicety.
        max_new_tokens=1536,
        tier=1,
        notes="Long-CoT reasoner; REQUIRES a large token budget or accuracy is an artifact.",
        aliases=("math_rl",),
        local_assembly="models/Instella-3B-Math-hf",
    ),
)

# Off-trajectory. Kept only because full-precision gradient attribution (TracIn) on a 1B
# model is the one thing this checkpoint is genuinely good for. Its accuracy comparison
# against Instella-3B is confounded (different data, different token counts) *and* its
# zero-shot output is degenerate, so it is excluded from the headline model axis.
OFF_TRAJECTORY: tuple[Checkpoint, ...] = (
    Checkpoint(
        tag="olmo1b",
        hf_id="amd/AMD-OLMo-1B",
        step=-1,
        intervention="unrelated 1B open model (different data + token budget)",
        is_base=True,
        saw_gsm8k_derived_data=False,
        max_new_tokens=768,
        tier=3,
        notes="Attribution-validation only; comparison to Instella-3B is confounded.",
        aliases=("olmo",),
    ),
)

ALL_CHECKPOINTS: tuple[Checkpoint, ...] = INSTELLA_TRAJECTORY + OFF_TRAJECTORY

_BY_KEY: dict[str, Checkpoint] = {}
for _ckpt in ALL_CHECKPOINTS:
    _BY_KEY[_ckpt.tag] = _ckpt
    _BY_KEY[_ckpt.hf_id] = _ckpt
    _BY_KEY[_ckpt.hf_id.split("/")[-1]] = _ckpt
    for _alias in _ckpt.aliases:
        _BY_KEY[_alias] = _ckpt


def resolve(name: str) -> Checkpoint:
    """Look up a checkpoint by tag, alias, HF id, or bare model name."""
    key = name.strip()
    found = _BY_KEY.get(key) or _BY_KEY.get(key.lower())
    if found is None:
        raise KeyError(
            f"Unknown checkpoint {name!r}. Known: {sorted({c.tag for c in ALL_CHECKPOINTS})}"
        )
    return found


def try_resolve(name: str) -> Checkpoint | None:
    """Same as :func:`resolve` but returns None instead of raising."""
    try:
        return resolve(name)
    except KeyError:
        return None


def tier(max_tier: int = 1, include_off_trajectory: bool = False) -> list[Checkpoint]:
    """Checkpoints at or below ``max_tier``, in trajectory order."""
    pool = ALL_CHECKPOINTS if include_off_trajectory else INSTELLA_TRAJECTORY
    return sorted((c for c in pool if c.tier <= max_tier), key=lambda c: c.step)


def transitions(checkpoints: list[Checkpoint] | None = None) -> list[tuple[Checkpoint, Checkpoint]]:
    """Consecutive (before, after) pairs along the trajectory.

    Each pair is a controlled data intervention; the pair
    ``(stage1, stage2)`` is the one that adds GSM8K-derived data and therefore carries
    the study's causal contamination contrast.
    """
    ordered = sorted(checkpoints or INSTELLA_TRAJECTORY, key=lambda c: c.step)
    return list(zip(ordered, ordered[1:], strict=False))


def contamination_intervention() -> tuple[Checkpoint, Checkpoint]:
    """The (control, treated) pair across which GSM8K-derived data is introduced."""
    before = [c for c in INSTELLA_TRAJECTORY if not c.saw_gsm8k_derived_data]
    after = [c for c in INSTELLA_TRAJECTORY if c.saw_gsm8k_derived_data]
    if not before or not after:  # pragma: no cover - registry invariant
        raise RuntimeError("trajectory has no GSM8K-data intervention boundary")
    return max(before, key=lambda c: c.step), min(after, key=lambda c: c.step)
