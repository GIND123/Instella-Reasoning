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
from instella_reasoning.records import (
    GenerationRecord,
    read_benchmark,
    read_corpus,
    read_jsonl,
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
    generate.add_argument("--batch-size", type=int, default=1)
    generate.add_argument("--no-cot-prompt", action="store_true", help="Disable CoT prompt wrapping.")

    score = subparsers.add_parser("score-generations", help="Score generation JSONL against benchmark answers.")
    score.add_argument("--benchmark", required=True)
    score.add_argument("--generations", required=True)
    score.add_argument("--output", required=True)

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

    report = subparsers.add_parser("report", help="Write a Markdown reliability report.")
    report.add_argument("--scores", required=True)
    report.add_argument("--contamination", required=True)
    report.add_argument("--output", required=True)

    train = subparsers.add_parser("train-command", help="Print an upstream Instella torchrun command.")
    train.add_argument("--config", required=True)

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
        rows = generate_with_transformers(
            benchmark=read_benchmark(args.benchmark),
            model_name_or_path=args.model,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            trust_remote_code=not args.no_trust_remote_code,
            load_in_4bit=args.load_in_4bit,
            batch_size=args.batch_size,
            use_cot_prompt=not args.no_cot_prompt,
        )
        print(f"Wrote {len(rows)} generations to {args.output}")
        return 0

    if args.command == "score-generations":
        generations = [GenerationRecord.from_dict(row) for row in read_jsonl(args.generations)]
        rows = score_generations(read_benchmark(args.benchmark), generations, args.output)
        print(f"Wrote {len(rows)} scored generations to {args.output}")
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

        results = compute_accuracy_gap(
            scores=load_evaluations(args.scores),
            contamination=load_contamination_hits(args.contamination),
            per_benchmark=not args.no_per_benchmark,
            per_model=not args.no_per_model,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps([result.to_dict() for result in results], indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(results)} accuracy-gap results to {args.output}")
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

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
