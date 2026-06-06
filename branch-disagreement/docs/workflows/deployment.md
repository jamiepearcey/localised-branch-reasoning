# Deployment Workflow (vast.ai over SSH)

This project runs its real experiment on a rented GPU (vast.ai). The local
machine only orchestrates; all model inference happens on the remote box.

## One-time setup

1. Rent an instance on vast.ai. A single 24-48 GB GPU (RTX 4090 / A6000 / L40S)
   is enough for a 7B-14B model. Choose a CUDA + PyTorch base image.
2. vast.ai gives you an SSH command like `ssh -p 41234 root@123.45.67.89`.
3. Copy `deploy/config.env.example` to `deploy/config.env` and fill in:
   - `SSH_HOST`, `SSH_PORT`, `SSH_USER` (usually `root`)
   - `SSH_KEY` (path to the private key you registered with vast.ai)
   - `REMOTE_DIR` (where code lives on the box, default `~/branch-disagreement`)
   - `MODEL` (HF id, e.g. `Qwen/Qwen2.5-14B-Instruct`)
   - `NLI_MODEL` (default `microsoft/deberta-large-mnli`)
   - run knobs: `DATASET`, `LIMIT`, `N_BRANCHES`, `TEMPERATURE`

`config.env` is git-ignored; it holds connection details and is never committed.

## End-to-end run

From `deploy/`:

```bash
./check_connection.sh   # verify SSH + GPU is visible (nvidia-smi)
./deploy.sh             # rsync prototype/ to REMOTE_DIR
./vast_setup.sh         # remote: venv + GPU deps + pre-download models
./run_remote.sh         # remote: run the experiment in tmux, log to file
./fetch_results.sh      # rsync results back into prototype/reports/
```

`run_remote.sh` starts the job inside a `tmux` session named `bdis` so it
survives SSH drops. Re-attach with:

```bash
./ssh_in.sh             # opens an interactive shell on the box
tmux attach -t bdis     # watch the run
```

## Notes

- All scripts read `deploy/config.env`; there are no hard-coded hosts.
- The base image varies by host. `vast_setup.sh` creates an isolated venv and
  pins what it can, but if vLLM fails to build, fall back to the host's existing
  torch and install `vllm` matching that CUDA version.
- Nothing here destroys the remote box. Stopping / destroying the instance is
  done from the vast.ai console so billing is explicit.
