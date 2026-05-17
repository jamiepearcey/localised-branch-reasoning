from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
import time
from pathlib import Path
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import pandas as pd

from localised_reasoning.coding_branch_eval import (
    CODING_BRANCH_PREFIX_CACHE_ID,
    CodingBranchPoint,
    CodingCase,
    CodingConsideration,
    LiveCodingBranchEngine,
    ProxyCodingBranchEngine,
    build_coding_branch_eval,
    render_coding_static_prefix,
)
from localised_reasoning.qa_scenarios import export_qa_csv


DATASETS_SERVER = "https://datasets-server.huggingface.co"
SWEBENCH_DATASETS = {
    "lite": "SWE-bench/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
}


@dataclass(frozen=True)
class SWEBenchInstance:
    instance_id: str
    dataset: str
    split: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    version: str
    difficulty: str
    patch_file_paths: tuple[str, ...]
    fail_to_pass: str
    pass_to_pass: str


def load_swebench_instances(
    *,
    dataset: str = "lite",
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
    shuffle_seed: int | None = None,
) -> list[SWEBenchInstance]:
    dataset_name = _dataset_name(dataset)
    if shuffle_seed is not None and limit is not None:
        total = _swebench_total_rows(dataset_name, split)
        rng = random.Random(shuffle_seed)
        selected_offsets = sorted(rng.sample(range(max(0, offset), total), min(limit, total - max(0, offset))))
        rows = _fetch_swebench_rows_by_offsets(dataset_name=dataset_name, split=split, offsets=selected_offsets)
    else:
        rows = _fetch_swebench_rows(dataset_name=dataset_name, split=split, offset=offset, length=limit or 100)
    return [_row_to_instance(row, dataset=dataset, split=split) for row in rows[:limit]]


def build_swebench_coding_cases(
    instances: Sequence[SWEBenchInstance],
    *,
    max_files: int = 2,
    max_file_chars: int = 9000,
    source_mode: str = "github-raw",
) -> tuple[list[CodingCase], pd.DataFrame]:
    cases: list[CodingCase] = []
    context_rows: list[dict[str, object]] = []
    for instance in instances:
        source_files: list[tuple[str, str]] = []
        if source_mode == "github-raw":
            for path in instance.patch_file_paths[:max_files]:
                source = fetch_base_file(instance.repo, instance.base_commit, path)
                if source is not None:
                    source_files.append((path, _truncate_middle(source, max_file_chars)))
        case = swebench_instance_to_coding_case(instance, source_files=source_files)
        cases.append(case)
        context_rows.append(
            {
                "instance_id": instance.instance_id,
                "dataset": instance.dataset,
                "split": instance.split,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "version": instance.version,
                "difficulty": instance.difficulty,
                "oracle_file_paths": ",".join(instance.patch_file_paths),
                "context_file_paths": ",".join(path for path, _ in source_files),
                "context_file_count": len(source_files),
                "problem_statement": instance.problem_statement,
                "fail_to_pass": instance.fail_to_pass,
                "pass_to_pass": instance.pass_to_pass,
            }
        )
    return cases, pd.DataFrame(context_rows)


def swebench_instance_to_coding_case(
    instance: SWEBenchInstance,
    *,
    source_files: Sequence[tuple[str, str]],
) -> CodingCase:
    code_context = render_swebench_code_context(instance, source_files)
    return CodingCase(
        case_id=instance.instance_id,
        language="python",
        title=f"{instance.repo} {instance.instance_id}",
        task=(
            "Produce a minimal SWE-bench patch for the issue. Use the issue text and base-commit source context only. "
            "Do not assume the gold patch. Final output must be a unified git diff."
        ),
        code=code_context,
        branch_points=(
            CodingBranchPoint(
                "method_start",
                "Issue contract and reproduction boundary",
                "problem statement, expected behavior, and first relevant source entry point",
                "The model must resolve what behavior is actually wrong before proposing edits.",
            ),
            CodingBranchPoint(
                "pre_mutation",
                "Candidate edit boundary",
                "the smallest source location likely to change",
                "This is where the patch can become over-broad or alter existing behavior.",
            ),
            CodingBranchPoint(
                "external_call",
                "Cross-module and API compatibility boundary",
                "calls, imports, public APIs, data structures, and downstream callers",
                "Repository fixes often fail because a local edit violates surrounding API assumptions.",
            ),
            CodingBranchPoint(
                "method_end",
                "Regression and final diff boundary",
                "tests implied by the issue and final unified diff shape",
                "The patch must be minimal, valid as a git diff, and aimed at fail-to-pass behavior without regressions.",
            ),
        ),
        considerations=default_swebench_considerations(),
        expected_branch_points=("method_start", "pre_mutation", "external_call", "method_end"),
    )


def render_swebench_code_context(instance: SWEBenchInstance, source_files: Sequence[tuple[str, str]]) -> str:
    files = "\n\n".join(
        f"### FILE: {path}\n"
        f"### Source outline\n{_source_outline(source)}\n"
        f"```python\n{_number_source_lines(source)}\n```"
        for path, source in source_files
    )
    if not files:
        files = "No source files were fetched. Use the issue text and file path hints only."
    return (
        f"### SWE-bench instance\n"
        f"instance_id: {instance.instance_id}\n"
        f"repo: {instance.repo}\n"
        f"base_commit: {instance.base_commit}\n"
        f"version: {instance.version}\n"
        f"oracle_file_paths_for_context_only: {', '.join(instance.patch_file_paths)}\n\n"
        f"### Issue\n{instance.problem_statement.strip()}\n\n"
        f"### Hints\n{instance.hints_text.strip() or 'None'}\n\n"
        f"### Base commit source context\n{files}\n"
    )


def default_swebench_considerations() -> tuple[CodingConsideration, ...]:
    return (
        CodingConsideration("source_localization", "Source localization", "Select the exact function or helper most likely responsible, using issue behavior and source line evidence.", ("function", "line", "helper", "source", "location")),
        CodingConsideration("contract", "Issue contract", "Restate the exact expected behavior and identify the narrow invariant to preserve.", ("expected", "actual", "invariant", "contract", "behavior")),
        CodingConsideration("edge_cases", "Regression edge cases", "Identify boundary examples and related existing behavior that should not change.", ("edge", "case", "regression", "existing", "boundary")),
        CodingConsideration("state_consistency", "Minimal patch shape", "Find the smallest edit that changes the bug without broad rewrites.", ("minimal", "diff", "change", "line", "patch")),
        CodingConsideration("security", "Compatibility risk", "Check imports, public APIs, types, and caller compatibility.", ("api", "type", "caller", "compat", "import")),
        CodingConsideration("tests", "Fail/pass validation", "Name specific tests or assertions implied by the issue.", ("test", "assert", "fail", "pass", "repro")),
    )


class LiveSWEBenchBranchEngine(LiveCodingBranchEngine):
    def final_from_checkpoints(self, case: CodingCase, checkpoint_collapses: Mapping[str, str]) -> str:
        response = self.worker.generate(
            render_swebench_final_patch_prompt(case, checkpoint_collapses),
            max_new_tokens=self.final_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _extract_patch(str(response.get("text", "")))

    def monolithic_review(self, case: CodingCase) -> str:
        response = self.worker.generate(
            render_swebench_monolithic_patch_prompt(case),
            max_new_tokens=self.final_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _extract_patch(str(response.get("text", "")))


class ProxySWEBenchBranchEngine(ProxyCodingBranchEngine):
    def final_from_checkpoints(self, case: CodingCase, checkpoint_collapses: Mapping[str, str]) -> str:
        return _proxy_patch(case)

    def monolithic_review(self, case: CodingCase) -> str:
        return _proxy_patch(case)


def render_swebench_final_patch_prompt(case: CodingCase, checkpoint_collapses: Mapping[str, str]) -> str:
    collapses = "\n\n".join(f"CHECKPOINT_COLLAPSE: {label}\n{text}" for label, text in checkpoint_collapses.items())
    return (
        "<|im_start|>system\n"
        "You are solving a SWE-bench instance. Create a minimal unified git diff from completed localized checkpoint collapses. "
        "Do not explain. Do not include markdown fences. Output only a patch beginning with diff --git. "
        "If the source context is insufficient, still output the smallest plausible diff against one shown file.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Task: {case.task}\n"
        f"Context:\n{case.code}\n\n"
        f"{collapses}\n\n"
        "Return only the model_patch string for SWE-bench JSONL.\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_swebench_monolithic_patch_prompt(case: CodingCase) -> str:
    return (
        "<|im_start|>system\n"
        "You are solving a SWE-bench instance in one uninterrupted reasoning stream. "
        "Output only a minimal unified git diff beginning with diff --git. Do not include markdown fences or explanation.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Task: {case.task}\nContext:\n{case.code}\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_swebench_branch_eval(
    *,
    engine,
    instances: Sequence[SWEBenchInstance],
    max_files: int = 2,
    max_file_chars: int = 9000,
    source_mode: str = "github-raw",
    consideration_limit: int = 5,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases, instance_df = build_swebench_coding_cases(
        instances,
        max_files=max_files,
        max_file_chars=max_file_chars,
        source_mode=source_mode,
    )
    case_df, branch_df, collapse_df, final_df, monolithic_df, summary_df = build_coding_branch_eval(
        engine=engine,
        cases=cases,
        consideration_limit=consideration_limit,
        progress_callback=progress_callback,
    )
    patch_df = build_patch_summary(final_df=final_df, monolithic_df=monolithic_df)
    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame(
                [
                    {"metric": "swebench_context_source_mode", "value": source_mode},
                    {"metric": "swebench_avg_context_file_count", "value": float(instance_df["context_file_count"].mean()) if len(instance_df) else 0},
                    {"metric": "branch_patch_valid_rate", "value": _valid_patch_rate(final_df, "answer")},
                    {"metric": "monolithic_patch_valid_rate", "value": _valid_patch_rate(monolithic_df, "answer")},
                ]
            ),
        ],
        ignore_index=True,
    )
    return instance_df, case_df, branch_df, collapse_df, final_df, monolithic_df, patch_df, summary_df


def export_swebench_branch_eval(
    *,
    output_xlsx: Path,
    instance_df: pd.DataFrame,
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    final_df: pd.DataFrame,
    monolithic_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Path:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        instance_df.to_excel(writer, sheet_name="SWE-bench Instances", index=False)
        case_df.to_excel(writer, sheet_name="Branch Plans", index=False)
        branch_df.to_excel(writer, sheet_name="Checkpoint Branches", index=False)
        collapse_df.to_excel(writer, sheet_name="Checkpoint Collapses", index=False)
        patch_df.to_excel(writer, sheet_name="Patch Summary", index=False)
        final_df.to_excel(writer, sheet_name="Branch Patches", index=False)
        monolithic_df.to_excel(writer, sheet_name="Monolithic Patches", index=False)
    return output_xlsx


def write_swebench_outputs(
    *,
    output_dir: Path,
    instance_df: pd.DataFrame,
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    final_df: pd.DataFrame,
    monolithic_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_qa_csv(summary_df, output_dir / "swebench_summary.csv")
    export_qa_csv(instance_df, output_dir / "swebench_instances.csv")
    export_qa_csv(case_df, output_dir / "swebench_branch_plans.csv")
    export_qa_csv(branch_df, output_dir / "swebench_checkpoint_branches.csv")
    export_qa_csv(collapse_df, output_dir / "swebench_checkpoint_collapses.csv")
    export_qa_csv(patch_df, output_dir / "swebench_patch_summary.csv")
    branch_predictions = output_dir / "swebench_branch_predictions.jsonl"
    monolithic_predictions = output_dir / "swebench_monolithic_predictions.jsonl"
    write_predictions_jsonl(final_df, branch_predictions, model_name=model_name)
    write_predictions_jsonl(monolithic_df, monolithic_predictions, model_name=f"{model_name}-monolithic")
    return branch_predictions, monolithic_predictions


def write_predictions_jsonl(df: pd.DataFrame, output_path: Path, *, model_name: str) -> Path:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(
                json.dumps(
                    {
                        "instance_id": row["case_id"],
                        "model_name_or_path": model_name,
                        "model_patch": row.get("answer", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path


def build_patch_summary(*, final_df: pd.DataFrame, monolithic_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, df in (("multi_checkpoint_branch", final_df), ("monolithic", monolithic_df)):
        for row in df.to_dict(orient="records"):
            patch = str(row.get("answer", ""))
            rows.append(
                {
                    "case_id": row.get("case_id", ""),
                    "method": method,
                    "patch_valid_shape": patch.lstrip().startswith("diff --git"),
                    "patch_chars": len(patch),
                    "file_count": len(parse_patch_file_paths(patch)),
                    "model_patch": patch,
                }
            )
    return pd.DataFrame(rows)


def parse_patch_file_paths(patch: str) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)\s*$", patch, flags=re.MULTILINE):
        path = match.group(2).strip()
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def fetch_base_file(repo: str, base_commit: str, path: str) -> str | None:
    encoded_path = "/".join(quote(part) for part in path.split("/"))
    url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{encoded_path}"
    try:
        with urlopen(url, timeout=45) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError):
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("utf-8", errors="replace")


def _dataset_name(dataset: str) -> str:
    if dataset in SWEBENCH_DATASETS:
        return SWEBENCH_DATASETS[dataset]
    if "/" in dataset:
        return dataset
    raise ValueError(f"unknown SWE-bench dataset: {dataset}")


def _fetch_swebench_rows(*, dataset_name: str, split: str, offset: int, length: int) -> list[dict[str, object]]:
    params = urlencode(
        {
            "dataset": dataset_name,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    payload = _urlopen_json(f"{DATASETS_SERVER}/rows?{params}")
    return [item["row"] for item in payload.get("rows", [])]


def _fetch_swebench_rows_by_offsets(*, dataset_name: str, split: str, offsets: Sequence[int], page_size: int = 100) -> list[dict[str, object]]:
    rows_by_offset: dict[int, dict[str, object]] = {}
    pages = sorted({offset // page_size * page_size for offset in offsets})
    for page in pages:
        for index, row in enumerate(_fetch_swebench_rows(dataset_name=dataset_name, split=split, offset=page, length=page_size)):
            absolute_offset = page + index
            if absolute_offset in offsets:
                rows_by_offset[absolute_offset] = row
    return [rows_by_offset[offset] for offset in offsets if offset in rows_by_offset]


def _swebench_total_rows(dataset_name: str, split: str) -> int:
    params = urlencode({"dataset": dataset_name, "config": "default"})
    payload = _urlopen_json(f"{DATASETS_SERVER}/size?{params}")
    for split_info in payload["size"]["splits"]:
        if split_info["split"] == split:
            return int(split_info["num_rows"])
    raise ValueError(f"SWE-bench split not found: {split}")


def _row_to_instance(row: dict[str, object], *, dataset: str, split: str) -> SWEBenchInstance:
    patch = str(row.get("patch", ""))
    return SWEBenchInstance(
        instance_id=str(row["instance_id"]),
        dataset=dataset,
        split=split,
        repo=str(row["repo"]),
        base_commit=str(row["base_commit"]),
        problem_statement=str(row.get("problem_statement", "")),
        hints_text=str(row.get("hints_text", "")),
        version=str(row.get("version", "")),
        difficulty=str(row.get("difficulty", "")),
        patch_file_paths=parse_patch_file_paths(patch),
        fail_to_pass=str(row.get("FAIL_TO_PASS", "")),
        pass_to_pass=str(row.get("PASS_TO_PASS", "")),
    )


def _urlopen_json(url: str) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n# ... truncated for context budget ...\n" + text[-tail:]


def _source_outline(source: str) -> str:
    rows: list[str] = []
    for index, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if re.match(r"(async\s+def|def|class)\s+\w+", stripped):
            rows.append(f"L{index}: {stripped}")
    return "\n".join(rows[:80]) or "No Python def/class outline found."


def _number_source_lines(source: str) -> str:
    return "\n".join(f"{index:04d}: {line}" for index, line in enumerate(source.splitlines(), start=1))


def _extract_patch(text: str) -> str:
    cleaned = re.sub(r"<think>\s*</think>", "", text, flags=re.I | re.S).strip()
    fence = re.search(r"```(?:diff|patch)?\s*(diff --git.+?)```", cleaned, flags=re.I | re.S)
    if fence:
        return fence.group(1).strip()
    start = cleaned.find("diff --git")
    if start >= 0:
        return cleaned[start:].strip()
    return cleaned


def _proxy_patch(case: CodingCase) -> str:
    path_match = re.search(r"oracle_file_paths_for_context_only:\s*([^\n,]+)", case.code)
    path = path_match.group(1).strip() if path_match else "example.py"
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,1 +1,1 @@\n"
        "-# proxy placeholder\n"
        "+# proxy placeholder\n"
    )


def _valid_patch_rate(df: pd.DataFrame, column: str) -> float:
    if not len(df):
        return 0.0
    return float(df[column].map(lambda value: str(value).lstrip().startswith("diff --git")).mean())


__all__ = [
    "CODING_BRANCH_PREFIX_CACHE_ID",
    "LiveSWEBenchBranchEngine",
    "ProxySWEBenchBranchEngine",
    "SWEBenchInstance",
    "build_swebench_branch_eval",
    "build_swebench_coding_cases",
    "export_swebench_branch_eval",
    "load_swebench_instances",
    "parse_patch_file_paths",
    "render_coding_static_prefix",
    "render_swebench_final_patch_prompt",
    "render_swebench_monolithic_patch_prompt",
    "write_swebench_outputs",
]
