from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import threading
import uuid
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class WorkerBranch:
    label: str
    marker: str


class LiveLlamaWorker:
    """Persistent JSONL client for the llama.cpp branch worker.

    The model is loaded once by the worker process. Each request gets fresh KV
    state inside that resident context, and branch requests do the prefix
    prefill/fork/marker/decode cycle without spawning a new model process.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        worker_path: Path = Path("build/llama_branch_worker"),
        ctx_size: int = 4096,
        gpu_layers: int = 999,
        batch_size: int = 512,
        max_seqs: int = 16,
        seed: int = 1234,
        startup_timeout_s: float = 180.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.worker_path = Path(worker_path)
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.batch_size = batch_size
        self.max_seqs = max_seqs
        self.seed = seed
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [
                str(self.worker_path),
                "--model",
                str(self.model_path),
                "--ctx-size",
                str(ctx_size),
                "--batch-size",
                str(batch_size),
                "--gpu-layers",
                str(gpu_layers),
                "--max-seqs",
                str(max_seqs),
                "--seed",
                str(seed),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_tail: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self.request({"cmd": "status"}, timeout_s=startup_timeout_s)

    def request(self, payload: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        if self._proc.poll() is not None:
            raise RuntimeError(f"llama branch worker exited with code {self._proc.returncode}: {self.stderr_tail}")
        request_payload = dict(payload)
        request_payload.setdefault("request_id", uuid.uuid4().hex)
        line = json.dumps(request_payload, ensure_ascii=False)
        with self._lock:
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
            response_line = self._readline(timeout_s=timeout_s)
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"worker returned non-JSON output: {response_line!r}") from exc
        if not response.get("ok"):
            raise RuntimeError(f"worker request failed: {response.get('error', response)}")
        return response

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        stop: str = "",
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self.request(
            {
                "cmd": "generate",
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "stop": stop,
            },
            timeout_s=timeout_s,
        )

    def cache_prefix(
        self,
        *,
        prefix_id: str,
        prefix: str,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self.request(
            {
                "cmd": "cache_prefix",
                "prefix_id": prefix_id,
                "prefix": prefix,
            },
            timeout_s=timeout_s,
        )

    def cached_generate(
        self,
        *,
        prefix_id: str,
        suffix: str,
        max_new_tokens: int = 128,
        stop: str = "",
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self.request(
            {
                "cmd": "cached_generate",
                "prefix_id": prefix_id,
                "suffix": suffix,
                "max_new_tokens": max_new_tokens,
                "stop": stop,
            },
            timeout_s=timeout_s,
        )

    def branch(
        self,
        *,
        prefix: str,
        branches: Sequence[WorkerBranch],
        max_new_tokens: int = 96,
        stop: str = "STOP_POINT:",
        parallel: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self.request(
            {
                "cmd": "branch",
                "prefix": prefix,
                "branches": [{"label": branch.label, "marker": branch.marker} for branch in branches],
                "max_new_tokens": max_new_tokens,
                "stop": stop,
                "parallel": parallel,
            },
            timeout_s=timeout_s,
        )

    def cached_branch(
        self,
        *,
        prefix_id: str,
        suffix: str,
        branches: Sequence[WorkerBranch],
        max_new_tokens: int = 96,
        stop: str = "STOP_POINT:",
        parallel: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self.request(
            {
                "cmd": "cached_branch",
                "prefix_id": prefix_id,
                "suffix": suffix,
                "branches": [{"label": branch.label, "marker": branch.marker} for branch in branches],
                "max_new_tokens": max_new_tokens,
                "stop": stop,
                "parallel": parallel,
            },
            timeout_s=timeout_s,
        )

    @property
    def stderr_tail(self) -> str:
        return "".join(self._stderr_tail[-80:])

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self.request({"cmd": "shutdown"}, timeout_s=10.0)
        except Exception:
            self._proc.terminate()
        finally:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)

    def __enter__(self) -> LiveLlamaWorker:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_tail.append(line)
            if len(self._stderr_tail) > 200:
                del self._stderr_tail[:100]

    def _readline(self, *, timeout_s: float | None) -> str:
        assert self._proc.stdout is not None
        if timeout_s is None:
            line = self._proc.stdout.readline()
        else:
            timer = threading.Timer(timeout_s, self._proc.kill)
            timer.start()
            try:
                line = self._proc.stdout.readline()
            finally:
                timer.cancel()
        if not line:
            raise RuntimeError(f"worker produced no response: {self.stderr_tail}")
        return line.rstrip("\n")
