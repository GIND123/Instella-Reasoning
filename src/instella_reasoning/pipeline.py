"""End-to-end orchestrator: one config -> the whole Reasoning Reliability Atlas.

``run_pipeline`` chains every stage the study needs so a fresh clone can produce the
full set of artifacts with a single command (``instella-reasoning run-all``):

1. (optional) expand the benchmark into consistency clusters (perturbations)
2. contamination scan (embedding + 13-gram, or lexical) -> ``contamination.jsonl``
3. (optional) model generation -> ``generations.jsonl`` (skipped if a generations
   file is supplied or no model is configured)
4. score generations -> ``scores.jsonl``
5. contaminated-vs-clean accuracy gap -> ``accuracy_gap.json``
6. embedding attribution -> ``attribution.jsonl`` (+ per-skill aggregate)
7. build the Reliability Atlas -> ``atlas.json``
8. write the full Markdown report -> ``report.md``
9. (optional) render all figures -> ``figures/``

Every stage is guarded and degrades gracefully: on a CPU-only machine with no model
and the dependency-free embedder, steps 2 and 4-9 still run on precomputed
generations (e.g. the bundled ``examples/``). Heavy stages announce what extra to
install if their dependency is missing, and are skipped rather than fatal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from instella_reasoning.analysis.accuracy_gap import compute_accuracy_gap
from instella_reasoning.analysis.atlas import build_atlas
from instella_reasoning.attribution_embedding import aggregate_by_skill, run_embedding_attribution
from instella_reasoning.contamination import (
    EmbeddingContaminationThresholds,
    run_contamination_scan,
    run_embedding_contamination_scan,
)
from instella_reasoning.evaluation import score_generations
from instella_reasoning.perturbations import PerturbationConfig, expand_benchmark_file
from instella_reasoning.quality import write_quality_report
from instella_reasoning.records import (
    GenerationRecord,
    read_benchmark,
    read_corpus,
    read_jsonl,
)
from instella_reasoning.reporting import write_full_report


@dataclass(slots=True)
class PipelineConfig:
    project_name: str = "instella-reasoning-atlas"
    seed: int = 6198
    output_dir: str = "outputs/atlas_run"

    benchmark: str = "examples/mini_benchmark.jsonl"
    corpus: str = "examples/mini_corpus.jsonl"
    generations: str | None = None  # precomputed generations; if None + model set, generate

    expand_variants: bool = False
    perturbations: tuple[str, ...] = (
        "entity_substitution",
        "premise_reordering",
        "irrelevant_context",
        "rephrasing",
    )
    numeric_variants: int = 0  # GSM-Symbolic-style answer-changing variants per item

    contamination_method: str = "embedding"  # or "lexical"
    contamination_top_k: int = 20
    embedder_backend: str = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_type: str = "flat"
    index_backend: str = "auto"

    model: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.0
    load_in_4bit: bool = False
    batch_size: int = 1

    attribution_top_k: int = 20
    near_duplicate_cosine: float = 0.85

    make_plots: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PipelineConfig:
        # Accept both a flat dict and the nested "sections" layout used in full.yaml.
        flat: dict[str, Any] = {}
        data = raw.get("data", {})
        cont = raw.get("contamination", {})
        gen = raw.get("generation", {})
        attr = raw.get("attribution", {})
        out = raw.get("outputs", {})

        flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
        flat.update(
            {
                "benchmark": data.get("benchmark", flat.get("benchmark", cls.benchmark)),
                "corpus": data.get("corpus", flat.get("corpus", cls.corpus)),
                "generations": data.get("generations", flat.get("generations")),
                "expand_variants": data.get("expand_variants", flat.get("expand_variants", False)),
                "perturbations": tuple(
                    data.get("perturbations", flat.get("perturbations", cls.perturbations))
                ),
                "numeric_variants": data.get("numeric_variants", flat.get("numeric_variants", 0)),
                "contamination_method": cont.get("method", flat.get("contamination_method", "embedding")),
                "contamination_top_k": cont.get("top_k", flat.get("contamination_top_k", 20)),
                "embedder_backend": cont.get("embedder_backend", flat.get("embedder_backend", "auto")),
                "embedding_model": cont.get("embedding_model", flat.get("embedding_model", cls.embedding_model)),
                "index_type": cont.get("index_type", flat.get("index_type", "flat")),
                "index_backend": cont.get("index_backend", flat.get("index_backend", "auto")),
                "model": gen.get("model", flat.get("model")),
                "max_new_tokens": gen.get("max_new_tokens", flat.get("max_new_tokens", 512)),
                "temperature": gen.get("temperature", flat.get("temperature", 0.0)),
                "load_in_4bit": gen.get("load_in_4bit", flat.get("load_in_4bit", False)),
                "batch_size": gen.get("batch_size", flat.get("batch_size", 1)),
                "attribution_top_k": attr.get("top_k", flat.get("attribution_top_k", 20)),
                "near_duplicate_cosine": attr.get(
                    "near_duplicate_cosine", flat.get("near_duplicate_cosine", 0.85)
                ),
                "make_plots": raw.get("plots", flat.get("make_plots", False)),
                "output_dir": out.get("dir", flat.get("output_dir", cls.output_dir)),
            }
        )
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in flat.items() if k in known})


def _log(step: str, message: str) -> None:
    print(f"[{step}] {message}", flush=True)


def run_pipeline(config: PipelineConfig) -> dict[str, str]:
    """Run the configured pipeline and return a map of stage -> artifact path."""
    from dataclasses import asdict

    from instella_reasoning.provenance import write_manifest

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    # Provenance first: git commit, package versions, and the resolved config, so every
    # artifact in this directory is traceable to exactly how it was produced.
    manifest_path = out / "manifest.json"
    manifest = write_manifest(manifest_path, config=asdict(config))
    artifacts["manifest"] = str(manifest_path)
    _log("provenance", f"run manifest -> {manifest_path}")

    # -- Stage 1: consistency-cluster expansion ------------------------------
    benchmark_path = config.benchmark
    if config.expand_variants:
        expanded = out / "benchmark_expanded.jsonl"
        count = expand_benchmark_file(
            config.benchmark,
            expanded,
            PerturbationConfig(
                types=tuple(config.perturbations),
                seed=config.seed,
                numeric_variants=config.numeric_variants,
            ),
        )
        benchmark_path = str(expanded)
        artifacts["benchmark"] = benchmark_path
        _log("variants", f"expanded to {count} items -> {expanded}")

    benchmark = read_benchmark(benchmark_path)
    corpus = read_corpus(config.corpus)
    _log("load", f"{len(benchmark)} benchmark items, {len(corpus)} corpus docs")

    # -- Stage 2: contamination scan -----------------------------------------
    contamination_path = out / "contamination.jsonl"
    if config.contamination_method == "embedding":
        hits = run_embedding_contamination_scan(
            benchmark=benchmark,
            corpus=corpus,
            output_path=contamination_path,
            top_k=config.contamination_top_k,
            embedder_backend=config.embedder_backend,
            embedding_model=config.embedding_model,
            index_type=config.index_type,
            index_backend=config.index_backend,
            thresholds=EmbeddingContaminationThresholds(),
        )
    else:
        hits = run_contamination_scan(benchmark, corpus, contamination_path, top_k=config.contamination_top_k)
    artifacts["contamination"] = str(contamination_path)
    _log("contamination", f"{len(hits)} hits -> {contamination_path}")

    # -- Stage 3: generation (optional) --------------------------------------
    generations_path = config.generations
    if generations_path is None and config.model:
        generations_path = str(out / "generations.jsonl")
        generations = _generate(config, benchmark, generations_path)
        if generations is None:
            generations_path = None
        else:
            artifacts["generations"] = generations_path
            _log("generate", f"{len(generations)} generations -> {generations_path}")
    elif generations_path:
        _log("generate", f"using precomputed generations {generations_path}")
    else:
        _log("generate", "no model and no generations file; skipping generation/scoring")

    # -- Stage 4-9: analysis on scores ---------------------------------------
    atlas_report = gap_results = skill_attributions = quality_report = None
    stratified_gap = None
    scores: list = []
    if generations_path and Path(generations_path).exists():
        scores_path = out / "scores.jsonl"
        generations = [GenerationRecord.from_dict(row) for row in read_jsonl(generations_path)]

        # Quality gate: flag degenerate completions before they reach the Atlas.
        quality_path = out / "quality.jsonl"
        quality_report = write_quality_report(generations, quality_path)
        artifacts["quality"] = str(quality_path)
        if quality_report.n_degenerate:
            _log(
                "quality",
                f"WARNING: {quality_report.n_degenerate}/{quality_report.n} completions look "
                f"degenerate ({quality_report.degenerate_fraction:.0%}) -> {quality_path}. "
                "Likely a prompt/format issue (e.g. missing chat template); scores are unreliable.",
            )
        else:
            _log("quality", f"{quality_report.n} completions OK -> {quality_path}")

        scores = score_generations(benchmark, generations, scores_path)
        artifacts["scores"] = str(scores_path)
        _log("score", f"{len(scores)} scored -> {scores_path}")

        gap_results = compute_accuracy_gap(scores, hits)
        gap_path = out / "accuracy_gap.json"
        gap_path.write_text(
            json.dumps([r.to_dict() for r in gap_results], indent=2), encoding="utf-8"
        )
        artifacts["accuracy_gap"] = str(gap_path)
        _log("accuracy-gap", f"{len(gap_results)} scopes -> {gap_path}")

        # Difficulty-adjusted, cluster-robust gap (confounder control; reviewer M3).
        from instella_reasoning.analysis.accuracy_gap import compute_stratified_accuracy_gap
        from instella_reasoning.difficulty import assign_difficulty_bins

        difficulty_bins = assign_difficulty_bins(benchmark, n_bins=3)
        stratified_gap = compute_stratified_accuracy_gap(scores, hits, difficulty_bins)
        strat_path = out / "accuracy_gap_stratified.json"
        strat_path.write_text(
            json.dumps([r.to_dict() for r in stratified_gap], indent=2), encoding="utf-8"
        )
        artifacts["accuracy_gap_stratified"] = str(strat_path)
        _log("accuracy-gap", f"difficulty-adjusted: {len(stratified_gap)} scopes -> {strat_path}")

        atlas_report = build_atlas(scores, hits)
        atlas_path = out / "atlas.json"
        atlas_path.write_text(json.dumps(atlas_report.to_dict(), indent=2), encoding="utf-8")
        artifacts["atlas"] = str(atlas_path)
        _log("atlas", f"{len(atlas_report.cells)} cells -> {atlas_path}")

    # -- attribution runs whether or not a model was available ----------------
    attribution_path = out / "attribution.jsonl"
    profiles = run_embedding_attribution(
        benchmark=benchmark,
        corpus=corpus,
        output_path=attribution_path,
        top_k=config.attribution_top_k,
        embedder_backend=config.embedder_backend,
        embedding_model=config.embedding_model,
        index_type=config.index_type,
        index_backend=config.index_backend,
        near_duplicate_cosine=config.near_duplicate_cosine,
    )
    skill_attributions = aggregate_by_skill(profiles)
    skill_agg_path = out / "attribution_by_skill.json"
    skill_agg_path.write_text(
        json.dumps([a.to_dict() for a in skill_attributions], indent=2), encoding="utf-8"
    )
    artifacts["attribution"] = str(attribution_path)
    artifacts["attribution_by_skill"] = str(skill_agg_path)
    _log("attribution", f"{len(profiles)} profiles -> {attribution_path}")

    # -- figures (optional) ---------------------------------------------------
    figure_paths: dict[str, str] = {}
    if config.make_plots:
        figure_paths = _render_figures(out, atlas_report, gap_results, skill_attributions, hits, scores)
        artifacts.update({f"figure_{name}": path for name, path in figure_paths.items()})

    # -- report ---------------------------------------------------------------
    report_path = out / "report.md"
    write_full_report(
        scores=scores,
        contamination=hits,
        output_path=report_path,
        atlas_report=atlas_report,
        gap_results=gap_results,
        skill_attributions=skill_attributions,
        figure_paths={name: _relative(path, out) for name, path in figure_paths.items()},
        quality_report=quality_report,
        stratified_gap_results=stratified_gap,
        manifest=manifest,
    )
    artifacts["report"] = str(report_path)
    _log("report", f"written -> {report_path}")
    return artifacts


def _generate(config: PipelineConfig, benchmark, output_path):
    from instella_reasoning.evaluation import generate_with_transformers

    try:
        return generate_with_transformers(
            benchmark=benchmark,
            model_name_or_path=config.model,
            output_path=output_path,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            load_in_4bit=config.load_in_4bit,
            batch_size=config.batch_size,
        )
    except RuntimeError as exc:
        _log("generate", f"skipped ({exc})")
        return None


def _render_figures(out, atlas_report, gap_results, skill_attributions, hits, scores):
    from instella_reasoning.analysis import plots

    figures_dir = out / "figures"
    produced: dict[str, str] = {}
    try:
        plots._require_matplotlib()
    except RuntimeError as exc:
        _log("plots", f"skipped ({exc})")
        return produced

    if atlas_report is not None and atlas_report.cells:
        produced["reliability_atlas"] = str(
            plots.plot_reliability_atlas(atlas_report, figures_dir / "reliability_atlas.png")
        )
        if atlas_report.clusters:
            produced["accuracy_consistency_quadrant"] = str(
                plots.plot_accuracy_consistency_quadrant(
                    atlas_report.clusters, figures_dir / "accuracy_consistency_quadrant.png"
                )
            )
    if gap_results:
        produced["accuracy_gap"] = str(plots.plot_accuracy_gap(gap_results, figures_dir / "accuracy_gap.png"))
    if hits:
        produced["contamination_breakdown"] = str(
            plots.plot_contamination_breakdown(hits, figures_dir / "contamination_breakdown.png")
        )
    if skill_attributions:
        produced["attribution_sources"] = str(
            plots.plot_attribution_sources(skill_attributions, figures_dir / "attribution_sources.png")
        )
    _log("plots", f"{len(produced)} figures -> {figures_dir}")
    return produced


def _relative(path: str, base: Path) -> str:
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return path
