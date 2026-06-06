# deploy/ — vast.ai SSH toolkit

Orchestrates the real experiment on a rented GPU. The local machine only drives;
all inference happens on the remote box. Full workflow notes:
[../docs/workflows/deployment.md](../docs/workflows/deployment.md).

## Setup

```bash
cp config.env.example config.env   # then edit: SSH_HOST/PORT/USER/KEY, MODEL, DATASET...
```

`config.env` is git-ignored — it holds your connection details.

## Scripts

| Script | What it does |
| --- | --- |
| `check_connection.sh` | Verify SSH works and `nvidia-smi` sees a GPU. |
| `deploy.sh` | Rsync `prototype/` to `REMOTE_DIR` (excludes caches/artifacts). |
| `vast_setup.sh` | Remote venv + GPU deps (vllm/transformers/datasets) + model prefetch. |
| `run_remote.sh` | Launch the experiment in a `tmux` session `bdis`, logging to `run.log`. |
| `status_remote.sh` | Report run status and tail the log (`status_remote.sh 80` for 80 lines). |
| `fetch_results.sh` | Rsync `reports/` back into local `prototype/reports/`. |
| `ssh_in.sh` | Open an interactive shell on the box (`tmux attach -t bdis` to watch). |

## Typical session

```bash
./check_connection.sh
./deploy.sh
./vast_setup.sh        # once per fresh box (a few minutes)
./run_remote.sh        # returns immediately; job runs in tmux
./status_remote.sh     # repeat until "no active 'bdis' session"
./fetch_results.sh     # pull the CSV/JSON back
```

Then read `prototype/reports/<OUTPUT_PREFIX>_summary.csv`: AUROC per detector,
DeLong p vs the branch-disagreement detector, and the per-question token cost.

## Notes

- Nothing here destroys the instance; stop/destroy from the vast.ai console so
  billing stays explicit.
- vast.ai base images vary. If `vllm` fails to install against the host torch,
  `vast_setup.sh` falls back to installing it standalone; if that also fails,
  pin `vllm` to the version matching the box's CUDA (e.g. `pip install vllm==0.6.3`).
- Re-running `deploy.sh` then `run_remote.sh` is the normal iterate loop. Re-run
  `vast_setup.sh` only when dependencies change.
