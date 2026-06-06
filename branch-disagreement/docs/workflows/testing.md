# Testing Workflow

## Expectations

- Run tests relevant to the files and behavior you changed.
- If you do not run tests, explain why in your handoff.
- Keep test additions proportional to risk and blast radius.

## Project notes

- The pure-logic test suite runs with **stdlib only** (no numpy / torch / vllm):

  ```bash
  cd prototype
  python3 -m unittest discover -s tests
  ```

- The CPU proxy smoke test exercises the full pipeline end to end with no model:

  ```bash
  cd prototype
  PYTHONPATH=src python3 scripts/smoke_test.py
  ```

- The GPU path (vLLM runner, NLI clustering) cannot be tested locally. Run it on
  the vast.ai box via `deploy/` and record results, or explain why not.
