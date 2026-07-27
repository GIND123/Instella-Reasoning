from __future__ import annotations

import argparse
from pathlib import Path

from instella_reasoning.attribution import (
    build_retrieval_attribution,
    load_contamination_hits,
    load_evaluations,
)
from instella_reasoning.contamination import (
    ContaminationThresholds,
    EmbeddingContaminationThresholds,
    run_contamination_scan,
    run_embedding_contamination_scan,
)
from instella_reasoning.evaluation import generate_with_transformers, score_generations
from instella_reasoning.perturbations import DEFAULT_PERTURBATIONS
from instella_reasoning.records import (
    GenerationRecord,
    read_benchmark,
    read_corpus,
    read_jsonl,
    write_jsonl,
)
from instella_reasoning.reporting import write_markdown_report
from instella_reasoning.training import TorchrunLaunchConfig, build_torchrun_command, shell_join


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="instella-reasoning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan-contamination", help="Search benchmark prompts against a corpus JSONL.")
    scan.add_argument("--benchmark", required=True)
    scan.add_argument("--corpus", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--top-k", type=int, default=5)
    scan.add_argument("--near-threshold", type=float, default=0.72)
    scan.add_argument("--candidate-threshold", type=float, default=0.45)

    escan = subparsers.add_parser(
        "scan-contamination-embedding",
        help="Embedding + 13-gram contamination scan producing C/PC/N labels.",
    )
    escan.add_argument("--benchmark", required=True)
    escan.add_argument("--corpus", required=True)
    escan.add_argument("--output", required=True)
    escan.add_argument("--top-k", type=int, default=20)
    escan.add_argument(
        "--embedder-backend",
        default="auto",
        choices=["auto", "sentence-transformers", "hashing"],
        help="auto uses sentence-transformers when installed, else a hashing fallback.",
    )
    escan.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    escan.add_argument("--index-type", default="flat", choices=["flat", "ivfpq", "hnsw"])
    escan.add_argument("--index-backend", default="auto", choices=["auto", "faiss", "bruteforce"])
    escan.add_argument("--contaminated-cosine", type=float, default=0.90)
    escan.add_argument("--partial-cosine", type=float, default=0.75)
    escan.add_argument("--contaminated-ngram", type=float, default=0.60)
    escan.add_argument("--partial-ngram", type=float, default=0.30)
    escan.add_argument("--ngram-n", type=int, default=13)

    loadbench = subparsers.add_parser(
        "load-benchmark", help="Download a reasoning benchmark from HuggingFace to JSONL."
    )
    loadbench.add_argument("--benchmark", required=True)
    loadbench.add_argument("--output", required=True)
    loadbench.add_argument("--split", default=None)
    loadbench.add_argument("--hf-name", default=None, help="Override the HF config (subject/subtask).")
    loadbench.add_argument("--limit", type=int, default=None)
    loadbench.add_argument(
        "--max-per-parent",
        type=int,
        default=None,
        help="Cap variants sharing a parent_id (e.g. GSM-Symbolic's 50 instances/template).",
    )

    loadcorpus = subparsers.add_parser(
        "load-corpus", help="Stream a HuggingFace corpus/dataset into CorpusDocument JSONL."
    )
    loadcorpus.add_argument("--hf-path", required=True)
    loadcorpus.add_argument("--output", required=True)
    loadcorpus.add_argument("--text-field", default="text")
    loadcorpus.add_argument("--source", default=None)
    loadcorpus.add_argument("--hf-name", default=None)
    loadcorpus.add_argument("--split", default="train")
    loadcorpus.add_argument("--limit", type=int, default=None)

    generate = subparsers.add_parser("generate", help="Generate model completions with Hugging Face Transformers.")
    generate.add_argument("--benchmark", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--no-trust-remote-code", action="store_true")
    generate.add_argument("--load-in-4bit", action="store_true", help="4-bit NF4 (CUDA + bitsandbytes).")
    generate.add_argument(
        "--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
        help="Weight dtype. auto=bf16. On a T4 (Turing) try fp16 for speed; ignored with --load-in-4bit.",
    )
    generate.add_argument(
        "--revision", default=None,
        help="Pin the model to a HF commit/tag/branch (reproducibility; silences remote-code re-downloads).",
    )
    generate.add_argument("--batch-size", type=int, default=1)
    generate.add_argument("--no-cot-prompt", action="store_true", help="Disable CoT prompt wrapping.")
    generate.add_argument(
        "--n-shot",
        type=int,
        default=0,
        help="Few-shot exemplars. Base checkpoints need >0 or they loop instead of answering.",
    )
    generate.add_argument(
        "--min-termination-rate",
        type=float,
        default=0.0,
        help="Abort if fewer than this share of outputs terminate naturally (0 = warn only).",
    )
    generate.add_argument(
        "--no-chat-template", action="store_true",
        help="Do not apply the tokenizer chat template (use only for base, non-Instruct models).",
    )
    generate.add_argument(
        "--limit", type=int, default=None,
        help="Only generate for the first N benchmark items (use for CPU/smoke checks).",
    )
    generate.add_argument(
        "--fail-degenerate", type=float, default=None, metavar="FRAC",
        help="Exit non-zero if more than FRAC of completions are degenerate (looping/empty). "
        "Use on the 4-item smoke to hard-stop a broken run before the full batch, e.g. 0.0.",
    )

    score = subparsers.add_parser("score-generations", help="Score generation JSONL against benchmark answers.")
    score.add_argument("--benchmark", required=True)
    score.add_argument("--generations", required=True)
    score.add_argument("--output", required=True)

    check = subparsers.add_parser(
        "check-generations",
        help="Flag degenerate (looping/empty) completions before scoring/atlas.",
    )
    check.add_argument("--generations", required=True)
    check.add_argument("--output", default=None, help="Optional JSONL of the flagged completions.")
    check.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="Exit non-zero if the degenerate fraction exceeds this (e.g. 0.1 for CI gating).",
    )

    attribute = subparsers.add_parser("attribute", help="Build retrieval-correctness attribution proxy rows.")
    attribute.add_argument("--contamination", required=True)
    attribute.add_argument("--scores", required=True)
    attribute.add_argument("--output", required=True)

    gap = subparsers.add_parser(
        "accuracy-gap",
        help="Compute contaminated-vs-clean accuracy gap (+ z-test) per benchmark/model.",
    )
    gap.add_argument("--scores", required=True)
    gap.add_argument("--contamination", required=True)
    gap.add_argument("--output", required=True)
    gap.add_argument("--no-per-benchmark", action="store_true")
    gap.add_argument("--no-per-model", action="store_true")
    gap.add_argument(
        "--benchmark", default=None,
        help="Benchmark JSONL; enables difficulty-stratified analysis (needs --stratified-output).",
    )
    gap.add_argument(
        "--stratified-output", default=None,
        help="Write a difficulty-adjusted, cluster-robust, MH-pooled gap here (needs --benchmark).",
    )
    gap.add_argument("--n-bins", type=int, default=3, help="Difficulty strata (equal-frequency).")

    report = subparsers.add_parser("report", help="Write a Markdown reliability report.")
    report.add_argument("--scores", required=True)
    report.add_argument("--contamination", required=True)
    report.add_argument("--output", required=True)

    train = subparsers.add_parser("train-command", help="Print an upstream Instella torchrun command.")
    train.add_argument("--config", required=True)

    variants = subparsers.add_parser(
        "make-variants", help="Expand a benchmark into semantics-preserving consistency clusters."
    )
    variants.add_argument("--benchmark", required=True)
    variants.add_argument("--output", required=True)
    variants.add_argument(
        "--types",
        nargs="+",
        default=list(DEFAULT_PERTURBATIONS),
        help="Perturbation types to apply (default: entity/reorder/irrelevant/rephrase).",
    )
    variants.add_argument("--seed", type=int, default=6198)
    variants.add_argument("--no-original", action="store_true", help="Exclude the original items.")
    variants.add_argument("--max-variants", type=int, default=None)
    variants.add_argument(
        "--numeric-k", type=int, default=0,
        help="Add K GSM-Symbolic-style answer-changing numeric variants per templatable item.",
    )

    validate = subparsers.add_parser(
        "validate-variants",
        help="Audit a variant suite: answer-preservation rate + text well-formedness (M2).",
    )
    validate.add_argument("--benchmark", required=True, help="A variant-suite JSONL (with originals).")
    validate.add_argument("--output", default=None, help="Optional JSON report path.")

    calib = subparsers.add_parser(
        "calibrate-contamination",
        help="Sweep the cosine threshold against a labeled set -> precision/recall/F1 (M4).",
    )
    calib.add_argument("--contamination", required=True)
    calib.add_argument(
        "--ground-truth", required=True,
        help="JSONL with {benchmark_id, contaminated: bool} rows of hand-labeled truth.",
    )
    calib.add_argument("--output", default=None)

    valrel = subparsers.add_parser(
        "validate-reliability",
        help="Check Reliability ranks genuine > fragile clusters on a labeled set (M5).",
    )
    valrel.add_argument("--scores", required=True)
    valrel.add_argument(
        "--labels", required=True,
        help="JSONL with {parent_id, status: 'genuine'|'fragile'} ground-truth rows.",
    )
    valrel.add_argument("--output", default=None)

    attr_emb = subparsers.add_parser(
        "attribute-embedding", help="Tier-1 embedding attribution: characterize each item's neighbors."
    )
    attr_emb.add_argument("--benchmark", required=True)
    attr_emb.add_argument("--corpus", required=True)
    attr_emb.add_argument("--output", required=True)
    attr_emb.add_argument("--by-skill", default=None, help="Optional path for the per-skill aggregate JSON.")
    attr_emb.add_argument("--top-k", type=int, default=20)
    attr_emb.add_argument("--embedder-backend", default="auto", choices=["auto", "sentence-transformers", "hashing"])
    attr_emb.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    attr_emb.add_argument("--index-type", default="flat", choices=["flat", "ivfpq", "hnsw"])
    attr_emb.add_argument("--index-backend", default="auto", choices=["auto", "faiss", "bruteforce"])
    attr_emb.add_argument("--near-duplicate-cosine", type=float, default=0.85)

    atlas = subparsers.add_parser("atlas", help="Build the Reasoning Reliability Atlas from scores + contamination.")
    atlas.add_argument("--scores", required=True)
    atlas.add_argument("--contamination", required=True)
    atlas.add_argument("--output", required=True, help="Atlas JSON output path.")
    atlas.add_argument("--markdown", default=None, help="Optional Markdown table output path.")
    atlas.add_argument("--skill-key", default="skill")

    emergence = subparsers.add_parser(
        "emergence", help="Scale/post-training emergence analysis across model score files."
    )
    emergence.add_argument("--small-scores", required=True)
    emergence.add_argument("--large-scores", required=True)
    emergence.add_argument("--contamination", required=True)
    emergence.add_argument("--output", required=True)
    emergence.add_argument("--small-name", default="small")
    emergence.add_argument("--large-name", default="large")
    emergence.add_argument("--pre-rl-scores", default=None, help="Pre-RL scores (e.g. Instella-3B-Instruct).")
    emergence.add_argument("--post-rl-scores", default=None, help="Post-RL scores (e.g. Instella-3B-Math).")

    plots = subparsers.add_parser("plots", help="Render the Atlas figure set (needs the viz extra).")
    plots.add_argument("--scores", required=True)
    plots.add_argument("--contamination", required=True)
    plots.add_argument("--output-dir", required=True)

    runall = subparsers.add_parser("run-all", help="Run the full end-to-end pipeline from a YAML config.")
    runall.add_argument("--config", required=True)

    # -- full-scale memorisation study ------------------------------------------

    splits = subparsers.add_parser(
        "build-splits",
        help="Build the verified seen/unseen GSM8K arms (exact 13-gram containment).",
    )
    splits.add_argument("--train", required=True, help="GSM8K train JSONL (candidate seen arm).")
    splits.add_argument("--test", required=True, help="GSM8K test JSONL (candidate unseen arm).")
    splits.add_argument("--corpus", required=True, help="Training corpus JSONL to verify against.")
    splits.add_argument("--output", required=True, help="Combined arms JSONL.")
    splits.add_argument("--containment-output", default=None, help="Per-item evidence JSON.")
    splits.add_argument("--n-per-arm", type=int, default=250)
    splits.add_argument("--n-bins", type=int, default=3, help="Difficulty strata for matching.")
    splits.add_argument("--seen-threshold", type=float, default=0.80)
    splits.add_argument("--unseen-threshold", type=float, default=0.10)
    splits.add_argument("--seed", type=int, default=6198)

    resample = subparsers.add_parser(
        "make-resample-suite",
        help="Duplicate items n times as the decoding-noise control (use with --temperature>0).",
    )
    resample.add_argument("--benchmark", required=True)
    resample.add_argument("--output", required=True)
    resample.add_argument("--n-samples", type=int, default=5)
    resample.add_argument("--limit", type=int, default=None)

    templates = subparsers.add_parser(
        "export-templates",
        help="Export auto-derived numeric variants for hand verification (CSV).",
    )
    templates.add_argument("--variants", required=True)
    templates.add_argument("--output", required=True)
    templates.add_argument(
        "--per-template", type=int, default=2, help="Rows to review per parent item."
    )

    review = subparsers.add_parser(
        "apply-template-review",
        help="Drop variants whose parent a human marked bad in the reviewed CSV.",
    )
    review.add_argument("--variants", required=True)
    review.add_argument("--review", required=True, help="The reviewed CSV (verdict column).")
    review.add_argument("--output", required=True)
    review.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any template is unreviewed (the pre-run gate).",
    )

    termcheck = subparsers.add_parser(
        "check-termination",
        help="Audit a generations JSONL for truncation before its accuracy is believed.",
    )
    termcheck.add_argument("--generations", required=True, nargs="+")
    termcheck.add_argument("--output", default=None)
    termcheck.add_argument("--min-rate", type=float, default=0.85)

    memo = subparsers.add_parser(
        "memorization",
        help="Seen/unseen difference-in-differences + cluster-robust regression (headline).",
    )
    memo.add_argument("--scores", required=True, nargs="+")
    memo.add_argument("--output", required=True)
    memo.add_argument("--n-bootstrap", type=int, default=4000)

    figs = subparsers.add_parser("figures", help="Render the eight-figure publication set.")
    figs.add_argument("--scores", required=True, nargs="+")
    figs.add_argument("--analysis", required=True, help="memorization command output JSON.")
    figs.add_argument("--output-dir", required=True)
    figs.add_argument("--containment", default=None)
    figs.add_argument("--termination", default=None)
    figs.add_argument("--n-items", type=int, default=250)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan-contamination":
        thresholds = ContaminationThresholds(
            near_duplicate=args.near_threshold,
            paraphrase_candidate=args.candidate_threshold,
        )
        rows = run_contamination_scan(
            benchmark=read_benchmark(args.benchmark),
            corpus=read_corpus(args.corpus),
            output_path=args.output,
            top_k=args.top_k,
            thresholds=thresholds,
        )
        print(f"Wrote {len(rows)} contamination hits to {args.output}")
        return 0

    if args.command == "scan-contamination-embedding":
        thresholds = EmbeddingContaminationThresholds(
            contaminated_cosine=args.contaminated_cosine,
            partial_cosine=args.partial_cosine,
            contaminated_ngram=args.contaminated_ngram,
            partial_ngram=args.partial_ngram,
            ngram_n=args.ngram_n,
        )
        rows = run_embedding_contamination_scan(
            benchmark=read_benchmark(args.benchmark),
            corpus=read_corpus(args.corpus),
            output_path=args.output,
            top_k=args.top_k,
            embedder_backend=args.embedder_backend,
            embedding_model=args.embedding_model,
            index_type=args.index_type,
            index_backend=args.index_backend,
            thresholds=thresholds,
        )
        print(f"Wrote {len(rows)} contamination hits to {args.output}")
        return 0

    if args.command == "load-benchmark":
        from instella_reasoning.datasets import load_benchmark_to_jsonl

        count = load_benchmark_to_jsonl(
            benchmark=args.benchmark,
            output_path=args.output,
            split=args.split,
            hf_name=args.hf_name,
            limit=args.limit,
            max_per_parent=args.max_per_parent,
        )
        print(f"Wrote {count} benchmark items to {args.output}")
        return 0

    if args.command == "load-corpus":
        from instella_reasoning.datasets import load_corpus_dataset_to_jsonl

        count = load_corpus_dataset_to_jsonl(
            hf_path=args.hf_path,
            output_path=args.output,
            text_field=args.text_field,
            source=args.source,
            hf_name=args.hf_name,
            split=args.split,
            limit=args.limit,
        )
        print(f"Wrote {count} corpus documents to {args.output}")
        return 0

    if args.command == "generate":
        benchmark = read_benchmark(args.benchmark)
        if args.limit is not None:
            benchmark = benchmark[: args.limit]
        rows = generate_with_transformers(
            benchmark=benchmark,
            model_name_or_path=args.model,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            trust_remote_code=not args.no_trust_remote_code,
            load_in_4bit=args.load_in_4bit,
            batch_size=args.batch_size,
            use_cot_prompt=not args.no_cot_prompt,
            use_chat_template=not args.no_chat_template,
            dtype=args.dtype,
            revision=args.revision,
            n_shot=args.n_shot,
        )
        print(f"Wrote {len(rows)} generations to {args.output}")

        # Truncation gate. A long-CoT checkpoint that never reaches its final-answer
        # marker inside the token budget scores as a weak model; that is a measurement
        # failure, not a result, and it must stop the run rather than flow downstream.
        from instella_reasoning.evaluation import termination_report

        term = termination_report(rows, threshold=max(args.min_termination_rate, 0.0))
        print(
            f"[generate] terminated naturally: {term.termination_rate:.1%} · "
            f"'####' marker: {term.marker_rate:.1%} · median {term.median_chars} chars"
        )
        if args.min_termination_rate > 0 and not term.passes:
            print(
                "\n" + "!" * 72 + "\n"
                f"ABORT: only {term.termination_rate:.1%} of completions terminated naturally "
                f"(gate {args.min_termination_rate:.0%}).\n"
                "The model is being cut off mid-reasoning, so its accuracy would be an\n"
                "artifact of --max-new-tokens. Raise the budget for this checkpoint and\n"
                "regenerate. A better answer extractor does NOT fix this.\n" + "!" * 72
            )
            return 1

        # Auto degeneracy gate: a broken run (wrong chat template, transformers 5.x vs
        # Instella remote code) yields looping/empty text that scores as pure "wrong" and
        # silently corrupts the Atlas. Surface it right at generation time.
        from instella_reasoning.quality import assess_generations

        quality = assess_generations(rows)
        if quality.n_degenerate:
            print(
                "\n" + "!" * 72 + "\n"
                f"WARNING: {quality.n_degenerate}/{quality.n} completions "
                f"({quality.degenerate_fraction:.0%}) look DEGENERATE (looping/empty).\n"
                "This is almost always a formatting/version problem, not a reasoning result:\n"
                "  - Instruct model missing its chat template (do NOT pass --no-chat-template), or\n"
                "  - transformers 5.x vs Instella's remote code -> pin `transformers>=4.44,<5`.\n"
                "Do not score/atlas this run until the smoke output is coherent CoT.\n"
                + "!" * 72
            )
            for row in quality.flagged[:5]:
                print(f"  - {row.benchmark_id}: {', '.join(row.reasons)} (tokens={row.n_tokens})")
        else:
            print(f"[quality] all {quality.n} completions look coherent (no degeneracy flags).")

        if args.fail_degenerate is not None and quality.degenerate_fraction > args.fail_degenerate:
            print(
                f"FAIL: degenerate fraction {quality.degenerate_fraction:.0%} exceeds "
                f"--fail-degenerate {args.fail_degenerate:.0%}."
            )
            return 1
        return 0

    if args.command == "score-generations":
        generations = [GenerationRecord.from_dict(row) for row in read_jsonl(args.generations)]
        rows = score_generations(read_benchmark(args.benchmark), generations, args.output)
        print(f"Wrote {len(rows)} scored generations to {args.output}")
        return 0

    if args.command == "check-generations":
        from instella_reasoning.quality import write_quality_report

        generations = [GenerationRecord.from_dict(row) for row in read_jsonl(args.generations)]
        report = write_quality_report(generations, args.output)
        print(
            f"Checked {report.n} completions: {report.n_degenerate} degenerate "
            f"({report.degenerate_fraction:.1%})."
        )
        for row in report.flagged[:10]:
            print(f"  - {row.benchmark_id}: {', '.join(row.reasons)} (tokens={row.n_tokens})")
        if args.output:
            print(f"Flagged completions -> {args.output}")
        if args.fail_threshold is not None and report.degenerate_fraction > args.fail_threshold:
            print(
                f"FAIL: degenerate fraction {report.degenerate_fraction:.1%} exceeds "
                f"threshold {args.fail_threshold:.1%}."
            )
            return 1
        return 0

    if args.command == "attribute":
        rows = build_retrieval_attribution(
            contamination_hits=load_contamination_hits(args.contamination),
            evaluations=load_evaluations(args.scores),
            output_path=args.output,
        )
        print(f"Wrote {len(rows)} attribution rows to {args.output}")
        return 0

    if args.command == "accuracy-gap":
        import json

        from instella_reasoning.analysis.accuracy_gap import compute_accuracy_gap

        scores = load_evaluations(args.scores)
        contamination = load_contamination_hits(args.contamination)
        results = compute_accuracy_gap(
            scores=scores,
            contamination=contamination,
            per_benchmark=not args.no_per_benchmark,
            per_model=not args.no_per_model,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps([result.to_dict() for result in results], indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(results)} accuracy-gap results to {args.output}")

        if args.stratified_output:
            if not args.benchmark:
                parser.error("--stratified-output requires --benchmark (to estimate difficulty).")
            from instella_reasoning.analysis.accuracy_gap import compute_stratified_accuracy_gap
            from instella_reasoning.difficulty import assign_difficulty_bins

            bins = assign_difficulty_bins(read_benchmark(args.benchmark), n_bins=args.n_bins)
            stratified = compute_stratified_accuracy_gap(
                scores=scores,
                contamination=contamination,
                difficulty_bins=bins,
                per_benchmark=not args.no_per_benchmark,
                per_model=not args.no_per_model,
            )
            Path(args.stratified_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.stratified_output).write_text(
                json.dumps([r.to_dict() for r in stratified], indent=2), encoding="utf-8"
            )
            print(f"Wrote {len(stratified)} difficulty-adjusted gap results to {args.stratified_output}")
        return 0

    if args.command == "validate-variants":
        import json

        from instella_reasoning.perturbations import validate_variants

        report = validate_variants(read_benchmark(args.benchmark))
        print(
            f"Variants: {report.n_variants} | answer-preserving OK: {report.n_preserved_ok}/"
            f"{report.n_answer_preserving} (rate {report.answer_preservation_rate:.3f}) | "
            f"answer-changing: {report.n_answer_changing} | degenerate-text: {report.n_degenerate_text}"
        )
        for vid in report.violations[:10]:
            print(f"  answer-preservation violation: {vid}")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"Wrote validation report to {args.output}")
        return 0

    if args.command == "calibrate-contamination":
        import json

        from instella_reasoning.validation import best_threshold, calibrate_cosine_threshold

        ground_truth = {
            str(row.get("benchmark_id") or row.get("id")): bool(row.get("contaminated"))
            for row in read_jsonl(args.ground_truth)
        }
        points = calibrate_cosine_threshold(load_contamination_hits(args.contamination), ground_truth)
        best = best_threshold(points)
        print("cosine  precision  recall  F1")
        for p in points:
            mark = "  <- best F1" if best and p.threshold == best.threshold else ""
            print(f"{p.threshold:.2f}    {p.precision:.3f}      {p.recall:.3f}   {p.f1:.3f}{mark}")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps({"curve": [p.to_dict() for p in points],
                            "best": best.to_dict() if best else None}, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote calibration curve to {args.output}")
        return 0

    if args.command == "validate-reliability":
        import json

        from instella_reasoning.validation import validate_reliability_metric

        labels = {
            str(row.get("parent_id") or row.get("id")): str(row.get("status"))
            for row in read_jsonl(args.labels)
        }
        result = validate_reliability_metric(load_evaluations(args.scores), labels)
        print(
            f"Reliability separation (genuine - fragile): {result.separation:+.3f} "
            f"(genuine={result.genuine_reliability:.3f} n={result.n_genuine}, "
            f"fragile={result.fragile_reliability:.3f} n={result.n_fragile}) -> "
            f"{'separates' if result.separates else 'DOES NOT separate'}"
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            print(f"Wrote metric-validation report to {args.output}")
        return 0

    if args.command == "report":
        content = write_markdown_report(
            scores=load_evaluations(args.scores),
            contamination=load_contamination_hits(args.contamination),
            output_path=args.output,
        )
        print(f"Wrote report to {Path(args.output)} ({len(content)} bytes)")
        return 0

    if args.command == "train-command":
        config = TorchrunLaunchConfig.from_yaml(args.config)
        print(shell_join(build_torchrun_command(config)))
        return 0

    if args.command == "make-variants":
        from instella_reasoning.perturbations import PerturbationConfig, expand_benchmark_file

        config = PerturbationConfig(
            types=tuple(args.types),
            seed=args.seed,
            include_original=not args.no_original,
            max_variants=args.max_variants,
            numeric_variants=args.numeric_k,
        )
        count = expand_benchmark_file(args.benchmark, args.output, config)
        print(f"Wrote {count} benchmark items (originals + variants) to {args.output}")
        return 0

    if args.command == "attribute-embedding":
        import json

        from instella_reasoning.attribution_embedding import (
            aggregate_by_skill,
            run_embedding_attribution,
        )

        profiles = run_embedding_attribution(
            benchmark=read_benchmark(args.benchmark),
            corpus=read_corpus(args.corpus),
            output_path=args.output,
            top_k=args.top_k,
            embedder_backend=args.embedder_backend,
            embedding_model=args.embedding_model,
            index_type=args.index_type,
            index_backend=args.index_backend,
            near_duplicate_cosine=args.near_duplicate_cosine,
        )
        print(f"Wrote {len(profiles)} attribution profiles to {args.output}")
        if args.by_skill:
            aggregates = aggregate_by_skill(profiles)
            Path(args.by_skill).parent.mkdir(parents=True, exist_ok=True)
            Path(args.by_skill).write_text(
                json.dumps([a.to_dict() for a in aggregates], indent=2), encoding="utf-8"
            )
            print(f"Wrote {len(aggregates)} per-skill aggregates to {args.by_skill}")
        return 0

    if args.command == "atlas":
        import json

        from instella_reasoning.analysis.atlas import build_atlas

        report = build_atlas(
            scores=load_evaluations(args.scores),
            contamination=load_contamination_hits(args.contamination),
            skill_key=args.skill_key,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote atlas with {len(report.cells)} cells to {args.output}")
        if args.markdown:
            Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
            Path(args.markdown).write_text(report.to_markdown(), encoding="utf-8")
            print(f"Wrote atlas Markdown to {args.markdown}")
        return 0

    if args.command == "emergence":
        import json

        from instella_reasoning.analysis.atlas import build_atlas
        from instella_reasoning.emergence import (
            EmergenceReport,
            analyze_rl_effect,
            atlas_consistency_probed,
            capability_transitions,
            rl_generalization_summary,
        )

        contamination = load_contamination_hits(args.contamination)
        small_atlas = build_atlas(load_evaluations(args.small_scores), contamination)
        large_atlas = build_atlas(load_evaluations(args.large_scores), contamination)
        transitions = capability_transitions(
            small_atlas, large_atlas, args.small_name, args.large_name
        )
        atlases = [small_atlas, large_atlas]
        report = EmergenceReport(transitions=transitions)
        if args.pre_rl_scores and args.post_rl_scores:
            pre_atlas = build_atlas(load_evaluations(args.pre_rl_scores), contamination)
            post_atlas = build_atlas(load_evaluations(args.post_rl_scores), contamination)
            report.rl_effects = analyze_rl_effect(pre_atlas, post_atlas)
            report.rl_generalization = rl_generalization_summary(report.rl_effects)
            atlases.extend([pre_atlas, post_atlas])
        report.consistency_probed = atlas_consistency_probed(*atlases)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote emergence analysis ({len(transitions)} transitions) to {args.output}")
        return 0

    if args.command == "plots":
        from instella_reasoning.analysis.atlas import build_atlas
        from instella_reasoning.analysis.plots import (
            plot_accuracy_consistency_quadrant,
            plot_contamination_breakdown,
            plot_reliability_atlas,
        )

        contamination = load_contamination_hits(args.contamination)
        report = build_atlas(load_evaluations(args.scores), contamination)
        output_dir = Path(args.output_dir)
        produced = []
        if report.cells:
            produced.append(plot_reliability_atlas(report, output_dir / "reliability_atlas.png"))
        if report.clusters:
            produced.append(
                plot_accuracy_consistency_quadrant(
                    report.clusters, output_dir / "accuracy_consistency_quadrant.png"
                )
            )
        if contamination:
            produced.append(
                plot_contamination_breakdown(contamination, output_dir / "contamination_breakdown.png")
            )
        print(f"Wrote {len(produced)} figures to {output_dir}")
        return 0

    if args.command == "run-all":
        from instella_reasoning.pipeline import PipelineConfig, run_pipeline

        artifacts = run_pipeline(PipelineConfig.from_yaml(args.config))
        print("\nArtifacts:")
        for stage, path in artifacts.items():
            print(f"  {stage}: {path}")
        return 0

    if args.command == "build-splits":
        return _cmd_build_splits(args)

    if args.command == "make-resample-suite":
        from instella_reasoning.perturbations import make_resample_suite

        items = read_benchmark(args.benchmark)
        if args.limit is not None:
            items = items[: args.limit]
        suite = make_resample_suite(items, args.n_samples)
        write_jsonl(args.output, suite)
        print(f"Wrote {len(suite)} items ({len(items)} originals x {args.n_samples} copies) to {args.output}")
        return 0

    if args.command == "export-templates":
        return _cmd_export_templates(args)

    if args.command == "apply-template-review":
        return _cmd_apply_template_review(args)

    if args.command == "check-termination":
        return _cmd_check_termination(args)

    if args.command == "memorization":
        return _cmd_memorization(args)

    if args.command == "figures":
        return _cmd_figures(args)

    parser.error(f"Unhandled command: {args.command}")
    return 2


# -- full-scale study command bodies ------------------------------------------------


def _cmd_build_splits(args) -> int:
    import json

    from instella_reasoning.datasets.splits import (
        balance_report,
        build_seen_unseen_split,
        verify_containment,
    )
    from instella_reasoning.difficulty import assign_difficulty_bins
    from instella_reasoning.records import CorpusDocument, read_jsonl

    train = read_benchmark(args.train)
    test = read_benchmark(args.test)
    print(f"[splits] candidates: {len(train)} train (seen?) / {len(test)} test (unseen?)")

    def stream():
        for row in read_jsonl(args.corpus):
            yield CorpusDocument.from_dict(row)

    containment = verify_containment(
        [*train, *test],
        stream(),
        seen_threshold=args.seen_threshold,
        unseen_threshold=args.unseen_threshold,
    )
    bins = assign_difficulty_bins([*train, *test], n_bins=args.n_bins)
    split = build_seen_unseen_split(
        train, test, containment, n_per_arm=args.n_per_arm, difficulty_bins=bins, seed=args.seed
    )
    write_jsonl(args.output, split.all_items())

    balance = balance_report(split, bins)
    summary = {**split.summary(), "balance": balance}
    print(json.dumps(summary, indent=2))
    if args.containment_output:
        intended = {i.id: "seen" for i in train}
        intended.update({i.id: "unseen" for i in test})
        payload = [
            {**result.to_dict(), "intended_arm": intended.get(bid)}
            for bid, result in sorted(containment.items())
        ]
        Path(args.containment_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.containment_output).write_text(
            json.dumps({"summary": summary, "items": payload}, indent=2), encoding="utf-8"
        )
        print(f"Wrote containment evidence to {args.containment_output}")
    if not split.usable:
        # Loud, and non-zero: an empty split means every downstream stage would run on
        # nothing and the GPU budget would be spent producing no measurable contrast.
        print("\n" + "!" * 72)
        print("ABORT: the seen/unseen split is empty — no contrast can be measured.")
        print(f"  {split.diagnosis()}")
        print("!" * 72)
        return 1
    if not balance["balanced"]:
        print("WARNING: arms are not difficulty-balanced; the seen/unseen contrast is confounded.")
    print(f"Wrote {len(split.all_items())} arm items to {args.output}")
    return 0


def _cmd_export_templates(args) -> int:
    """Emit a CSV a human can read left-to-right and mark ok/bad, one row per variant."""
    import csv
    from collections import defaultdict

    items = read_benchmark(args.variants)
    originals = {i.id: i for i in items if i.variant_type == "original"}
    per_parent: dict[str, list] = defaultdict(list)
    for item in items:
        if item.variant_type == "gsm_symbolic":
            per_parent[item.parent_id].append(item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "parent_id", "verdict(ok|bad)", "original_question", "original_answer",
                "variant_question", "variant_answer", "magnitude_ratio", "reviewer_note",
            ]
        )
        for parent_id, variants in sorted(per_parent.items()):
            parent = originals.get(parent_id)
            for variant in variants[: args.per_template]:
                writer.writerow(
                    [
                        parent_id, "", " ".join((parent.prompt if parent else "").split()),
                        parent.answer if parent else "",
                        " ".join(variant.prompt.split()), variant.answer,
                        variant.metadata.get("magnitude_ratio", ""), "",
                    ]
                )
                n += 1
    print(
        f"Wrote {n} rows for {len(per_parent)} templates to {out}\n"
        "Fill the verdict column with ok/bad, then run apply-template-review."
    )
    return 0


def _cmd_apply_template_review(args) -> int:
    import csv

    items = read_benchmark(args.variants)
    verdicts: dict[str, str] = {}
    with Path(args.review).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parent = (row.get("parent_id") or "").strip()
            verdict = (row.get("verdict(ok|bad)") or row.get("verdict") or "").strip().lower()
            if not parent or not verdict:
                continue
            # Any 'bad' row condemns the whole template: one wrong instantiation means the
            # template's substitution rule is unsound, not just that instance.
            if verdicts.get(parent) != "bad":
                verdicts[parent] = verdict

    templated = {i.parent_id for i in items if i.variant_type == "gsm_symbolic"}
    unreviewed = sorted(templated - set(verdicts))
    if unreviewed:
        message = f"{len(unreviewed)} template(s) unreviewed: {unreviewed[:5]}"
        if args.require_complete:
            print(f"FAIL: {message}")
            return 1
        print(f"WARNING: {message}")

    bad = {p for p, v in verdicts.items() if v.startswith("b")}
    kept = [i for i in items if not (i.variant_type == "gsm_symbolic" and i.parent_id in bad)]
    write_jsonl(args.output, kept)
    print(
        f"Reviewed {len(verdicts)} templates, rejected {len(bad)}; "
        f"wrote {len(kept)}/{len(items)} items to {args.output}"
    )
    return 0


def _cmd_check_termination(args) -> int:
    import json

    from instella_reasoning.evaluation import termination_report
    from instella_reasoning.records import GenerationRecord, read_jsonl

    reports = []
    failed = False
    for path in args.generations:
        rows = [GenerationRecord.from_dict(r) for r in read_jsonl(path)]
        report = termination_report(rows, threshold=args.min_rate)
        reports.append({**report.to_dict(), "path": path})
        flag = "OK " if report.passes else "FAIL"
        failed = failed or not report.passes
        print(
            f"  [{flag}] {report.model:34s} n={report.n:5d} "
            f"terminated={report.termination_rate:6.1%} marker={report.marker_rate:6.1%} "
            f"median_chars={report.median_chars}"
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if failed:
        print(
            "\nAt least one model is being scored on truncated output. Raise --max-new-tokens\n"
            "for that checkpoint and regenerate; a better answer extractor will not fix this."
        )
        return 1
    return 0


def _read_scores(paths: list[str]):
    from instella_reasoning.records import EvaluationRecord, read_jsonl

    records = []
    for path in paths:
        for row in read_jsonl(path):
            records.append(
                EvaluationRecord(
                    benchmark_id=row["benchmark_id"],
                    parent_id=row.get("parent_id") or row["benchmark_id"],
                    variant_type=row.get("variant_type", "original"),
                    expected=row.get("expected"),
                    predicted=row.get("predicted", ""),
                    normalized_expected=row.get("normalized_expected"),
                    normalized_predicted=row.get("normalized_predicted", ""),
                    correct=bool(row.get("correct")),
                    model=row.get("model", "unknown"),
                    metadata=dict(row.get("metadata", {})),
                )
            )
    return records


def _cmd_memorization(args) -> int:
    import json

    from instella_reasoning.analysis.memorization import summarize

    records = _read_scores(args.scores)
    result = summarize(records, n_bootstrap=args.n_bootstrap)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nMemorisation analysis over {len(records)} scored generations\n")
    for model, payload in result["per_model"].items():
        did = payload["did"]
        if did["status"] != "measured":
            print(f"  {model:34s} {did['status']}")
            continue
        lo, hi = did["cluster_bootstrap_ci95"]
        print(
            f"  {model:34s} DiD={did['difference_in_differences']:+.3f} "
            f"CI=[{lo:+.3f},{hi:+.3f}] {'*' if did['excludes_zero'] else ' '}"
        )
        print(f"      {did['interpretation']}")
    print(f"\nWrote {args.output}")
    return 0


def _cmd_figures(args) -> int:
    import json

    from instella_reasoning.analysis.figures import render_all

    records = _read_scores(args.scores)
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    containment = None
    if args.containment:
        payload = json.loads(Path(args.containment).read_text(encoding="utf-8"))
        containment = payload.get("items", payload) if isinstance(payload, dict) else payload
    termination = None
    if args.termination:
        termination = json.loads(Path(args.termination).read_text(encoding="utf-8"))

    written = render_all(
        records, analysis, args.output_dir,
        containment=containment, termination=termination, n_items=args.n_items,
    )
    made = {k: v for k, v in written.items() if v}
    for name, path in sorted(made.items()):
        print(f"  {name}: {path}")
    missing = sorted(k for k, v in written.items() if not v)
    if missing:
        print(f"  (skipped, inputs unavailable: {', '.join(missing)})")
    print(f"Wrote {len(made)} figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
