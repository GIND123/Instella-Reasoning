#!/usr/bin/env python3
"""Assemble an HF-transformers-loadable local copy of amd/Instella-3B-Math.

Why: the Instella-3B-Math HF repo ships remote code written for the vLLM serving
engine (its modeling_instella.py unconditionally imports vllm), so transformers
cannot load it. But its checkpoint is weight-compatible with the HF-style modeling
code shipped by amd/Instella-3B-Instruct (verified: both index 399 tensors with
identical names and dimensions — same OLMo-2-style architecture, RL-finetuned).

This script downloads the Math weights/tokenizer/config, drops in the Instruct
repo's modeling_instella.py, and points config.json's auto_map at it. Generation
then uses the local directory as the model path:

    python experiments/fetch_instella_math_hf.py
    instella-reasoning generate --model models/Instella-3B-Math-hf ...

--no-weights builds config/tokenizer/modeling only (cheap smoke test of the
assembly + auto_map wiring on a machine that will never run the model).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

MATH_REPO = "amd/Instella-3B-Math"
INSTRUCT_REPO = "amd/Instella-3B-Instruct"
DEFAULT_OUT = "models/Instella-3B-Math-hf"
HF_AUTO_MAP = {
    "AutoConfig": "modeling_instella.InstellaConfig",
    "AutoModelForCausalLM": "modeling_instella.InstellaForCausalLM",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-weights", action="store_true",
                        help="Skip the safetensors download (assembly smoke test only).")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("error: needs huggingface_hub — pip install -e '.[hf]'", file=sys.stderr)
        return 1

    out = Path(args.out)
    marker = out / "modeling_instella.py"
    if marker.exists() and (out / "config.json").exists():
        if args.no_weights or any(out.glob("*.safetensors")):
            print(out)
            return 0

    ignore = ["*.py"]  # never take the Math repo's vLLM-only python files
    if args.no_weights:
        ignore.append("*.safetensors")
    snapshot_download(MATH_REPO, local_dir=out, ignore_patterns=ignore)

    modeling = hf_hub_download(INSTRUCT_REPO, "modeling_instella.py")
    shutil.copy(modeling, out / "modeling_instella.py")
    (out / "configuration_instella.py").unlink(missing_ok=True)

    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["auto_map"] = dict(HF_AUTO_MAP)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
