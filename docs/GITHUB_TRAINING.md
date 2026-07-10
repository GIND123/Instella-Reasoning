# GitHub Training Guide

Real Instella training requires GPU hardware. GitHub-hosted runners are useful
for tests, linting, and report generation, but they are not appropriate for
3B-scale ROCm training.

## Recommended Setup

Use a self-hosted runner with labels like:

```text
self-hosted, linux, x64, rocm, mi300x
```

For a full Instella-style pretraining run, attach the runner to a login node or
single worker that can launch `torchrun` across the cluster. The workflow in
`.github/workflows/amd-base-self-hosted.yml` is manually triggered so expensive
jobs do not start on every push.

## Required Secrets

- `HF_TOKEN`: only needed for private/gated Hugging Face resources.
- `WANDB_API_KEY`: optional; needed only when W&B logging is enabled.

## Upstream Checkout

The training script expects AMD's upstream repository at `external/Instella`.
The workflow clones it automatically when absent.

## Multi-Node Runs

Edit `configs/training/amd_base_multinode.yaml` or pass a different config when
dispatching the workflow. Each node must agree on:

- `num_nodes`
- `gpus_per_node`
- `job_id`
- `master_addr`
- `master_port`
- the per-node `node_rank`

For production clusters, prefer the scheduler's native node allocation and use
GitHub only as the orchestration/audit entrypoint.
