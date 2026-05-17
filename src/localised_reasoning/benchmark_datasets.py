from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
import time
from typing import Iterable, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from localised_reasoning.comparative_eval import EvalQuestion


MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_CONFIG = "default"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class BenchmarkInfo:
    source: str
    split: str
    row_count: int
    categories: tuple[str, ...]


def load_benchmark_questions(
    *,
    source: str,
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
    categories: Sequence[str] | None = None,
    shuffle_seed: int | None = None,
) -> list[EvalQuestion]:
    if source == "mmlu-pro":
        return load_mmlu_pro_questions(
            split=split,
            limit=limit,
            offset=offset,
            categories=categories,
            shuffle_seed=shuffle_seed,
        )
    raise ValueError(f"unknown benchmark source: {source}")


def benchmark_info(source: str, *, split: str = "test") -> BenchmarkInfo:
    if source != "mmlu-pro":
        raise ValueError(f"unknown benchmark source: {source}")
    rows = _mmlu_pro_total_rows(split)
    categories = sorted(
        {
            str(row.get("category", "uncategorized"))
            for row in _fetch_mmlu_pro_rows(split=split, offset=0, length=min(rows, 100))
        }
    )
    return BenchmarkInfo(source=source, split=split, row_count=rows, categories=tuple(categories))


def load_mmlu_pro_questions(
    *,
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
    categories: Sequence[str] | None = None,
    shuffle_seed: int | None = None,
) -> list[EvalQuestion]:
    local_rows = _load_mmlu_pro_local_rows(split)
    if local_rows is not None:
        return _select_mmlu_pro_rows(
            local_rows,
            limit=limit,
            offset=offset,
            categories=categories,
            shuffle_seed=shuffle_seed,
        )

    selected_categories = {category.strip().lower() for category in categories or () if category.strip()}

    if shuffle_seed is not None and not selected_categories and limit is not None:
        total = _mmlu_pro_total_rows(split)
        upper = max(0, total - max(0, offset))
        sample_size = min(limit, upper)
        rng = random.Random(shuffle_seed)
        selected_offsets = sorted(rng.sample(range(max(0, offset), total), sample_size))
        return [
            _mmlu_pro_row_to_question(row)
            for row in _fetch_mmlu_pro_rows_by_offsets(split=split, offsets=selected_offsets)
        ]

    rows = _iter_mmlu_pro_rows(split=split, offset=offset)
    if selected_categories:
        rows = (
            row
            for row in rows
            if str(row.get("category", "")).strip().lower() in selected_categories
        )

    if shuffle_seed is not None:
        buffered = list(rows)
        rng = random.Random(shuffle_seed)
        rng.shuffle(buffered)
        rows = iter(buffered)

    questions: list[EvalQuestion] = []
    for row in rows:
        questions.append(_mmlu_pro_row_to_question(row))
        if limit is not None and len(questions) >= limit:
            break
    return questions


def _iter_mmlu_pro_rows(*, split: str, offset: int = 0, page_size: int = 100) -> Iterable[dict[str, object]]:
    total = _mmlu_pro_total_rows(split)
    position = max(0, offset)
    while position < total:
        length = min(page_size, total - position)
        rows = _fetch_mmlu_pro_rows(split=split, offset=position, length=length)
        if not rows:
            break
        yield from rows
        position += length


def _fetch_mmlu_pro_rows(*, split: str, offset: int, length: int) -> list[dict[str, object]]:
    params = urlencode(
        {
            "dataset": MMLU_PRO_DATASET,
            "config": MMLU_PRO_CONFIG,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    payload = _urlopen_json(f"{DATASETS_SERVER}/rows?{params}")
    return [item["row"] for item in payload.get("rows", [])]


def _fetch_mmlu_pro_rows_by_offsets(*, split: str, offsets: Sequence[int], page_size: int = 100) -> list[dict[str, object]]:
    rows_by_offset: dict[int, dict[str, object]] = {}
    pages = sorted({offset // page_size * page_size for offset in offsets})
    for page in pages:
        for index, row in enumerate(_fetch_mmlu_pro_rows(split=split, offset=page, length=page_size)):
            absolute_offset = page + index
            if absolute_offset in offsets:
                rows_by_offset[absolute_offset] = row
    return [rows_by_offset[offset] for offset in offsets if offset in rows_by_offset]


def _load_mmlu_pro_local_rows(split: str) -> list[dict[str, object]] | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    filename = f"data/{split}-00000-of-00001.parquet"
    try:
        path = hf_hub_download(
            repo_id=MMLU_PRO_DATASET,
            repo_type="dataset",
            filename=filename,
        )
        df = pd.read_parquet(path)
    except Exception:
        return None
    return df.to_dict(orient="records")


def _select_mmlu_pro_rows(
    rows: Sequence[dict[str, object]],
    *,
    limit: int | None,
    offset: int,
    categories: Sequence[str] | None,
    shuffle_seed: int | None,
) -> list[EvalQuestion]:
    selected_categories = {category.strip().lower() for category in categories or () if category.strip()}
    selected_rows = list(rows)
    if selected_categories:
        selected_rows = [
            row
            for row in selected_rows
            if str(row.get("category", "")).strip().lower() in selected_categories
        ]
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(selected_rows)
    if offset:
        selected_rows = selected_rows[max(0, offset):]
    if limit is not None:
        selected_rows = selected_rows[:limit]
    return [_mmlu_pro_row_to_question(row) for row in selected_rows]


def _mmlu_pro_total_rows(split: str) -> int:
    params = urlencode({"dataset": MMLU_PRO_DATASET, "config": MMLU_PRO_CONFIG})
    payload = _urlopen_json(f"{DATASETS_SERVER}/size?{params}")
    for split_info in payload["size"]["splits"]:
        if split_info["split"] == split:
            return int(split_info["num_rows"])
    raise ValueError(f"MMLU-Pro split not found: {split}")


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


def _mmlu_pro_row_to_question(row: dict[str, object]) -> EvalQuestion:
    options = [str(option) for option in row["options"]]  # type: ignore[index]
    answer_index = int(row["answer_index"])  # type: ignore[arg-type]
    if answer_index < 0 or answer_index >= len(options):
        raise ValueError(f"invalid MMLU-Pro answer_index: {answer_index}")
    answer_label = OPTION_LABELS[answer_index]
    answer_text = options[answer_index]
    category = str(row.get("category") or "uncategorized")
    source = str(row.get("src") or category)
    question_id = f"mmlu-pro-{row['question_id']}"
    return EvalQuestion(
        question_id=question_id,
        category=f"mmlu-pro/{category}/{source}",
        question=_format_mmlu_pro_question(str(row["question"]), options),
        expected_answer=f"{answer_label}. {answer_text}",
        accepted_patterns=_multiple_choice_patterns(answer_label, answer_text),
    )


def _format_mmlu_pro_question(question: str, options: Sequence[str]) -> str:
    option_rows = "\n".join(
        f"{OPTION_LABELS[index]}. {option}"
        for index, option in enumerate(options)
    )
    return (
        f"{question.strip()}\n\n"
        f"Options:\n{option_rows}\n\n"
        "Answer with the option letter and the answer text."
    )


def _multiple_choice_patterns(answer_label: str, answer_text: str) -> tuple[str, ...]:
    escaped_label = re.escape(answer_label)
    stripped_answer = re.sub(r"\s+", " ", answer_text.strip())
    patterns = [
        rf"^\s*\(?{escaped_label}\)?(?:[.)\]:\s]|$)",
        rf"\banswer\s*(?:is|:)\s*\(?{escaped_label}\)?\b",
        rf"\boption\s+{escaped_label}\b",
        rf"\bchoice\s+{escaped_label}\b",
    ]
    if stripped_answer and stripped_answer.upper() != "N/A":
        patterns.append(rf"^\s*{re.escape(stripped_answer)}\b")
        patterns.append(rf"\b{re.escape(stripped_answer)}\b")
    return tuple(patterns)
