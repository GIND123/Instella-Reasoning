from __future__ import annotations

import argparse
from pathlib import Path

from instella_reasoning.attribution import (
    build_retrieval_attribution,
    load_contamination_hits,
    load_evaluations,
)
from instella_reasoning.contamination import ContaminationThresholds, run_contamination_scan
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

    generate = subparsers.add_parser("generate", help="Generate model completions with Hugging Face Transformers.")
    generate.add_argument("--benchmark", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--no-trust-remote-code", action="store_true")

    score = subparsers.add_parser("score-generations", help="Score generation JSONL against benchmark answers.")
    score.add_argument("--benchmark", required=True)
    score.add_argument("--generations", required=True)
    score.add_argument("--output", required=True)

    attribute = subparsers.add_parser("attribute", help="Build retrieval-correctness attribution proxy rows.")
    attribute.add_argument("--contamination", required=True)
    attribute.add_argument("--scores", required=True)
    attribute.add_argument("--output", required=True)

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

    if args.command == "generate":
        rows = generate_with_transformers(
            benchmark=read_benchmark(args.benchmark),
            model_name_or_path=args.model,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            trust_remote_code=not args.no_trust_remote_code,
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
