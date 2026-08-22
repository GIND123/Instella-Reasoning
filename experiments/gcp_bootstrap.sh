#!/usr/bin/env bash
# Provision a GCP Spot GPU box and run G1 (the quantity ablation) end to end.
#
# Differences from lambda_vllm_bootstrap.sh, which this reuses rather than duplicates:
#
#   * GCP images vary in whether the NVIDIA driver is present. We check rather than
#     assume, and fail with the exact install command instead of failing later inside
#     vLLM with an error that does not name the cause.
#   * Spot instances are preempted with ~30 seconds' notice. Everything below is
#     resumable: the item set comes from git, generations are skipped if already
#     present, and a background loop mirrors logs and results to the HF dataset every
#     LOG_SYNC_SEC so that a preemption loses at most that window. Without the loop the
#     only copy of a half-finished run dies with the VM.
#   * The run is launched under nohup and the script returns, so an ssh drop does not
#     kill the job.
#
# Requires HF_TOKEN in the environment (for both model downloads and the result sync).
#
#   export HF_TOKEN=hf_...
#   bash experiments/gcp_bootstrap.sh                    # provision + run
#   STEP=provision bash experiments/gcp_bootstrap.sh     # provision only
#   STEP=run       bash experiments/gcp_bootstrap.sh     # run only

set -euo pipefail

REPO="${REPO:-$HOME/instella-reasoning}"
BRANCH="${BRANCH:-agent/quantity-ablation}"
GIT_URL="${GIT_URL:-https://github.com/GIND123/Instella-Reasoning.git}"
HF_REPO="${HF_REPO:-GOVINDFROM/Instella-Reasoning}"
RUN_DIR="experiments/runs/qty-ablation-v1"
LOG_SYNC_SEC="${LOG_SYNC_SEC:-180}"
STEP="${STEP:-all}"

: "${HF_TOKEN:?set HF_TOKEN before running (export HF_TOKEN=hf_...)}"

provision() {
  echo "== [1/5] GPU check"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
No nvidia-smi on this box. Install the driver first, then re-run:

  # Debian/Ubuntu Deep Learning VM:
  sudo /opt/deeplearning/install-driver.sh
  # or, on a plain image:
  curl -fsSL -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
  sudo python3 install_gpu_driver.py
EOF
    exit 2
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  local mem
  mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
  if [[ "$mem" -lt 70000 ]]; then
    echo "WARNING: ${mem} MiB of VRAM. G1 is inference-only and fits in 40 GB," >&2
    echo "         but the injection runs (G2) need ~58 GB. Fine for G1." >&2
  fi

  echo "== [2/5] system packages"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3-venv python3-pip >/dev/null

  echo "== [3/5] repository @ $BRANCH"
  if [[ -d "$REPO/.git" ]]; then
    git -C "$REPO" fetch --quiet origin
    git -C "$REPO" checkout --quiet "$BRANCH"
    git -C "$REPO" reset --hard --quiet "origin/$BRANCH"
  else
    git clone --quiet --branch "$BRANCH" "$GIT_URL" "$REPO"
  fi
  git -C "$REPO" log --oneline -1

  echo "== [4/5] engines"
  bash "$REPO/experiments/lambda_bootstrap.sh"       # transformers venv + package
  bash "$REPO/experiments/lambda_vllm_bootstrap.sh"  # vLLM venv + out-of-tree arch

  echo "== [5/5] item set"
  local bench="$REPO/$RUN_DIR/base/qty_ablation.jsonl"
  if [[ ! -s "$bench" ]]; then
    echo "item set not in git; rebuilding from parents" >&2
    "$HOME/.venvs/instella/bin/python" "$REPO/experiments/build_quantity_ablation.py" \
      --parents "$REPO/experiments/runs/ckpt-axis-v1/base/gsm8k_train_parents.jsonl" \
                "$REPO/experiments/runs/ckpt-axis-v1/base/gsm8k_test_parents.jsonl" \
      --out-dir "$REPO/$RUN_DIR/base" --n-parents 1779 --seed 6198
  fi
  echo "   $(wc -l < "$bench") variant rows ready"
}

# Mirror logs + analysis to HF on a timer. Its own process so a preemption during
# generation still leaves the last window's output on the Hub.
start_log_sync() {
  cat > "$REPO/.log_sync.sh" <<EOF
#!/usr/bin/env bash
while true; do
  sleep $LOG_SYNC_SEC
  HF_INCLUDE_GENERATIONS=0 "\$HOME/.venvs/instella/bin/python" \\
    "$REPO/experiments/hf_sync.py" push --repo "$HF_REPO" \\
    --path "$RUN_DIR" --message "heartbeat \$(date -u +%H:%M:%SZ)" >/dev/null 2>&1 || true
done
EOF
  chmod +x "$REPO/.log_sync.sh"
  nohup "$REPO/.log_sync.sh" >/dev/null 2>&1 &
  echo "$!" > "$REPO/.log_sync.pid"
  echo "== log sync every ${LOG_SYNC_SEC}s (pid $(cat "$REPO/.log_sync.pid"))"
}

run() {
  mkdir -p "$REPO/$RUN_DIR/logs"
  start_log_sync
  echo "== launching G1 under nohup; tail $REPO/$RUN_DIR/logs/g1.log"
  cd "$REPO"
  REPO="$REPO" HF_REPO="$HF_REPO" nohup bash experiments/run_quantity_ablation.sh \
    > "$REPO/$RUN_DIR/logs/g1.log" 2>&1 &
  echo "$!" > "$REPO/.g1.pid"
  sleep 20
  tail -n 25 "$REPO/$RUN_DIR/logs/g1.log" || true
  echo
  echo "   pid $(cat "$REPO/.g1.pid"); results appear at"
  echo "   https://huggingface.co/datasets/$HF_REPO/tree/main/$RUN_DIR"
}

case "$STEP" in
  provision) provision ;;
  run)       run ;;
  all)       provision; run ;;
  *) echo "STEP must be provision|run|all" >&2; exit 2 ;;
esac
