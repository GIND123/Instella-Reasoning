from instella_reasoning.training import TorchrunLaunchConfig, build_torchrun_command


def test_build_torchrun_command_contains_upstream_paths() -> None:
    config = TorchrunLaunchConfig(upstream_dir="external/Instella", overrides=["--dry_run=true"])

    command = build_torchrun_command(config)

    assert command[0] == "torchrun"
    assert "external/Instella/scripts/train.py" in command
    assert "external/Instella/configs/instella-3b-pretrain-stage1.yaml" in command
    assert "--dry_run=true" in command
