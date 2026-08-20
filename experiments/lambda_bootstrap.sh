#!/usr/bin/env bash
# Provision a fresh Lambda Labs instance to match the Modal image exactly.
#
# Runs ON the box, after experiments/lambda_sync.sh has pushed the working tree.
# Every pin here mirrors the image definition in experiments/modal_fullscale.py:66.
# The transformers pin is exact and load-bearing for two reasons recorded there:
# Instella ships remote modeling code written against the 4.4x attention/cache_position
# API, which 5.x removed and which degrades the forward pass silently rather than
# erroring; and a minor-version drift between checkpoints enters the DiD as though it
# were a model difference.
#
#   ssh ubuntu@<ip> 'bash ~/instella-reasoning/experiments/lambda_bootstrap.sh'

set -euo pipefail

REPO="${REPO:-$HOME/instella-reasoning}"
VENV="${VENV:-$HOME/.venvs/instella}"
HF_CACHE="${HF_CACHE:-$HOME/hf}"

echo "==> system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl rsync

# Modal's image is python 3.11, but pyproject.toml requires only >=3.10 and Lambda Stack
# 22.04 ships 3.10 with no reliable 3.11 in its repos. Prefer 3.11 where it exists so the
# two environments match, and fall back rather than failing the box.
PY=""
for cand in python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { echo "no usable python found" >&2; exit 1; }
sudo apt-get install -y -qq "${PY}-venv" "${PY}-dev" || true
echo "    using $PY ($("$PY" --version 2>&1))"

echo "==> virtualenv at $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "==> python dependencies (pins mirror modal_fullscale.py)"
pip install --quiet \
    "transformers==4.56.0" \
    "torch>=2.3.0" \
    "accelerate>=0.30.0" \
    "datasets>=2.19.0" \
    "huggingface-hub>=0.23.0" \
    "safetensors>=0.4.3" \
    "sympy>=1.12" \
    "matplotlib>=3.7.0" \
    "scipy>=1.11.0" \
    "numpy>=1.24.0" \
    "pyyaml>=6.0.1" \
    "tqdm>=4.66.0"

echo "==> environment"
mkdir -p "$HF_CACHE"
cat > "$HOME/.instella_env" <<EOF
export HF_HOME="$HF_CACHE"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src"
export PATH="$VENV/bin:\$PATH"
EOF
grep -qxF 'source ~/.instella_env' "$HOME/.bashrc" || echo 'source ~/.instella_env' >> "$HOME/.bashrc"
# shellcheck disable=SC1091
source "$HOME/.instella_env"

echo "==> instella-reasoning console shim"
# Matches the Modal image's shim: avoids `pip install -e .` against a tree we do not
# want to make writable, and avoids rebuilding metadata on every start.
sudo tee /usr/local/bin/instella-reasoning >/dev/null <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" -c 'import sys; from instella_reasoning.cli import main; sys.exit(main())' "\$@"
EOF
sudo chmod +x /usr/local/bin/instella-reasoning

echo
echo "==> verification"
python - <<'PY'
import torch, transformers
assert transformers.__version__ == "4.56.0", f"WRONG transformers: {transformers.__version__}"
print(f"transformers {transformers.__version__}  torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name}  {p.total_memory / 1e9:.0f} GB")
PY
python -c "import instella_reasoning; print('package import OK')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "bootstrap complete. HF_TOKEN still needs exporting before any run:"
echo "  echo 'export HF_TOKEN=hf_...' >> ~/.instella_env && source ~/.instella_env"
