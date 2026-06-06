#!/usr/bin/env bash
# Launch the experiment on the box inside a tmux session ('bdis') so it survives
# SSH drops. Writes a self-contained run_job.sh on the remote (config values
# baked in) and runs it under tmux — avoids fragile nested shell quoting.
# Logs to $REMOTE_DIR/prototype/run.log. Returns immediately.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

: "${MODEL:?set MODEL in config.env}"
: "${DATASET:?set DATASET in config.env}"

echo "Starting run in tmux session 'bdis' on $(remote_target) ..."

# Config values are passed as env to the remote shell, which bakes them into
# run_job.sh. None of these contain spaces, so quoting stays simple.
remote_ssh "REMOTE_DIR='${REMOTE_DIR}' \
  MODEL='${MODEL}' \
  NLI_MODEL='${NLI_MODEL:-microsoft/deberta-large-mnli}' \
  DATASET='${DATASET}' \
  SPLIT='${SPLIT:-}' \
  RESPONSE_MODE='${RESPONSE_MODE:-short}' \
  LIMIT='${LIMIT:-500}' \
  N_BRANCHES='${N_BRANCHES:-8}' \
  TEMPERATURE='${TEMPERATURE:-0.8}' \
  MAX_NEW_TOKENS='${MAX_NEW_TOKENS:-32}' \
  BRANCH_MODE='${BRANCH_MODE:-self_consistency}' \
  OUTPUT_PREFIX='${OUTPUT_PREFIX:-run}' \
  bash -s" <<'REMOTE'
set -euo pipefail
cd "${REMOTE_DIR}/prototype"
command -v tmux >/dev/null 2>&1 || { apt-get update -y && apt-get install -y tmux; }
if tmux has-session -t bdis 2>/dev/null; then
  echo "A 'bdis' session is already running. Attach via deploy/ssh_in.sh then 'tmux attach -t bdis'." >&2
  exit 1
fi

# Write the launcher. Unquoted heredoc: config vars expand now (baked in);
# runtime expressions like $(date) are escaped to run when the job runs.
cat > run_job.sh <<JOB
#!/usr/bin/env bash
set -euo pipefail
cd "${REMOTE_DIR}/prototype"
source "${REMOTE_DIR}/.venv/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "== run started: \$(date) =="
PYTHONPATH=src python3 scripts/run_experiment.py \
  --engine vllm \
  --model "${MODEL}" \
  --nli-model "${NLI_MODEL}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --response-mode "${RESPONSE_MODE}" \
  --limit "${LIMIT}" \
  --n-branches "${N_BRANCHES}" \
  --temperature "${TEMPERATURE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --branch-mode "${BRANCH_MODE}" \
  --output-prefix "${OUTPUT_PREFIX}"
echo "== run finished: \$(date) exit=\$? =="
JOB
chmod +x run_job.sh

# tee both the job's stdout and a completion marker into run.log.
tmux new-session -d -s bdis "bash run_job.sh 2>&1 | tee run.log"
echo "Launched. Log: ${REMOTE_DIR}/prototype/run.log"
REMOTE

echo "Run started. Use deploy/status_remote.sh to watch, deploy/fetch_results.sh when done."
