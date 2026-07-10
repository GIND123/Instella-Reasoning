from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TorchrunLaunchConfig:
    upstream_dir: str = "external/Instella"
    script: str = "scripts/train.py"
    train_config: str = "configs/instella-3b-pretrain-stage1.yaml"
    num_nodes: int = 1
    gpus_per_node: int = 8
    node_rank: int | None = None
    job_id: str = "amd-base"
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    load_path: str | None = None
    overrides: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TorchrunLaunchConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(**payload)


def build_torchrun_command(config: TorchrunLaunchConfig) -> list[str]:
    command = [
        "torchrun",
        f"--nnodes={config.num_nodes}",
        f"--nproc_per_node={config.gpus_per_node}",
        f"--rdzv_id={config.job_id}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={config.master_addr}:{config.master_port}",
    ]
    if config.node_rank is not None:
        command.append(f"--node_rank={config.node_rank}")

    script_path = (Path(config.upstream_dir) / config.script).as_posix()
    train_config_path = (Path(config.upstream_dir) / config.train_config).as_posix()
    command.extend([script_path, train_config_path])

    if config.load_path:
        command.extend(["--load_path", config.load_path])
    command.extend(config.overrides)
    return command


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
