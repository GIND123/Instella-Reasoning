#!/usr/bin/env bash
# Provision the vLLM engine in a venv SEPARATE from the transformers one.
#
# vllm 0.8.5.post1 pins its own torch build. Installing it into ~/.venvs/instella would
# replace the torch that the transformers path was verified against, so the measurement
# that has already been made (Phase 0) could not be reproduced on this box afterwards.
# Two venvs, one pin each; the transformers env stays exactly as bootstrapped.
#
# Every pin below is copied from experiments/modal_reasoning.py:89, where this exact
# combination was established against these checkpoints:
#   * 0.6.3 rejects the architecture outright; the upstream vLLM-native class needs a
#     VllmConfig __init__ (post-0.6.3) and vllm.model_executor.layers.sampler.Sampler
#     (removed once V1 is mandatory). 0.8.5 is inside the window holding both.
#   * VLLM_USE_V1=0 for the same reason: the registered class implements the V0 sampler.
#   * The architecture module must sit on PYTHONPATH, not be injected via sys.path — the
#     engine inspects architectures in a subprocess that inherits the environment but not
#     the parent's sys.path edits.
#
#   ssh ubuntu@<ip> 'bash ~/instella-reasoning/experiments/lambda_vllm_bootstrap.sh'

set -euo pipefail

REPO="${REPO:-$HOME/instella-reasoning}"
VENV="${VENV:-$HOME/.venvs/vllm}"
ARCH_DIR="${ARCH_DIR:-$HOME/arch}"
HF_CACHE="${HF_CACHE:-$HOME/hf}"

PY=""
for cand in python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { echo "no usable python found" >&2; exit 1; }
echo "==> using $PY ($("$PY" --version 2>&1))"

echo "==> virtualenv at $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "==> vllm + pinned transformers (this pulls vllm's own torch; ~10 min)"
pip install --quiet "vllm==0.8.5.post1" "transformers==4.56.0" \
    "huggingface-hub>=0.23.0" "datasets>=2.19.0" "sympy>=1.12" "numpy>=1.24.0"

echo "==> out-of-tree architecture module"
mkdir -p "$ARCH_DIR"
HF_HOME="$HF_CACHE" python - "$ARCH_DIR" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
# Only the reasoning-specialised repo ships a vLLM-native implementation of the shared
# 3B architecture; the others ship a transformers-style module the engine cannot consume.
# Registering this one class therefore serves every 3B checkpoint.
src = hf_hub_download("amd/Instella-3B-Math", "modeling_instella.py")
dst = f"{sys.argv[1]}/vllm_arch.py"
shutil.copyfile(src, dst)
print(f"    {dst}")
PY

cat > "$HOME/.instella_vllm_env" <<EOF
export HF_HOME="$HF_CACHE"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_V1=0
export PYTHONPATH="$REPO/src:$ARCH_DIR"
export PATH="$VENV/bin:\$PATH"
EOF

echo
echo "==> verification"
# shellcheck disable=SC1091
source "$HOME/.instella_vllm_env"
python - <<'PY'
import torch, transformers, vllm
assert transformers.__version__ == "4.56.0", f"WRONG transformers: {transformers.__version__}"
print(f"vllm {vllm.__version__}  transformers {transformers.__version__}  torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
import vllm_arch
print(f"arch module OK: {vllm_arch.__file__}")
assert hasattr(vllm_arch, "InstellaForCausalLM"), "arch module has no InstellaForCausalLM"
import instella_reasoning
print("instella_reasoning import OK")
PY

echo
echo "vllm bootstrap complete. Use with:"
echo "  source ~/.instella_vllm_env && python experiments/vllm_generate.py ..."
