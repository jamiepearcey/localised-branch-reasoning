"""Dataset loading. Normalises sources into a common ``EvalQuestion`` schema.

The built-in ``sample`` set carries proxy hints (``p_known`` and ``distractors``)
that the CPU ``ProxyRunner`` uses to synthesise branch answers. Real datasets
(PopQA, TriviaQA) do not carry those hints; the real runner ignores them.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalQuestion:
    id: str
    question: str
    gold_answers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in sample set (offline; for proxy smoke tests and unit tests).
# `p_known` is the proxy's prior that the model has internalised the fact;
# `distractors` are wrong answers the proxy scatters across when it has not.
# `popularity` mimics PopQA's popularity axis.
# ---------------------------------------------------------------------------
_SAMPLE: List[Dict[str, Any]] = [
    {"q": "What is the capital of France?", "a": ["Paris"],
     "p_known": 0.98, "popularity": 0.99, "distractors": ["Lyon", "Marseille"]},
    {"q": "Who wrote the play 'Romeo and Juliet'?", "a": ["William Shakespeare", "Shakespeare"],
     "p_known": 0.95, "popularity": 0.97, "distractors": ["Christopher Marlowe", "Ben Jonson"]},
    {"q": "What is the chemical symbol for gold?", "a": ["Au"],
     "p_known": 0.9, "popularity": 0.9, "distractors": ["Ag", "Gd", "Go"]},
    {"q": "In what year did the first manned Moon landing occur?", "a": ["1969"],
     "p_known": 0.9, "popularity": 0.95, "distractors": ["1968", "1970", "1972"]},
    {"q": "What is the largest planet in the Solar System?", "a": ["Jupiter"],
     "p_known": 0.92, "popularity": 0.94, "distractors": ["Saturn", "Neptune"]},
    {"q": "Who painted the Mona Lisa?", "a": ["Leonardo da Vinci", "Leonardo", "da Vinci"],
     "p_known": 0.9, "popularity": 0.93, "distractors": ["Michelangelo", "Raphael"]},
    {"q": "What is the mayor of the Italian town of Vico Equense as of 2019?", "a": ["Andrea Buonocore"],
     "p_known": 0.08, "popularity": 0.05, "distractors": ["Giuseppe Russo", "Mario Esposito", "Luca Ferraro"]},
    {"q": "Who composed the soundtrack for the 1981 film 'The Aviator'?", "a": ["Dominic Frontiere"],
     "p_known": 0.07, "popularity": 0.04, "distractors": ["John Barry", "Jerry Goldsmith", "Bill Conti"]},
    {"q": "What is the population of the village of Llanfihangel-y-Creuddyn?", "a": ["around 600"],
     "p_known": 0.05, "popularity": 0.03, "distractors": ["1200", "350", "2100"]},
    {"q": "Who discovered the asteroid 1990 MU?", "a": ["Robert McNaught", "Rob McNaught"],
     "p_known": 0.06, "popularity": 0.04, "distractors": ["Carolyn Shoemaker", "Eleanor Helin", "David Levy"]},
    {"q": "What is the boiling point of water at sea level in Celsius?", "a": ["100", "100 degrees"],
     "p_known": 0.95, "popularity": 0.96, "distractors": ["90", "110", "212"]},
    {"q": "Who was the first president of the United States?", "a": ["George Washington", "Washington"],
     "p_known": 0.97, "popularity": 0.98, "distractors": ["Thomas Jefferson", "John Adams"]},
]


def _sample_questions(limit: Optional[int]) -> List[EvalQuestion]:
    rows = _SAMPLE if limit is None else _SAMPLE[:limit]
    out = []
    for i, r in enumerate(rows):
        out.append(
            EvalQuestion(
                id=f"sample-{i:03d}",
                question=r["q"],
                gold_answers=list(r["a"]),
                metadata={
                    "p_known": r["p_known"],
                    "popularity": r["popularity"],
                    "distractors": list(r["distractors"]),
                    "source": "sample",
                },
            )
        )
    return out


# Per-dataset default split. PopQA only ships a labelled `test` split; TriviaQA's
# `test` split has hidden answers, so labelled eval uses `validation`.
_DEFAULT_SPLIT = {"popqa": "test", "triviaqa": "validation"}


def load_dataset(
    name: str = "sample",
    limit: Optional[int] = None,
    split: Optional[str] = None,
    shuffle_seed: Optional[int] = None,
) -> List[EvalQuestion]:
    """Return a list of ``EvalQuestion``.

    ``sample`` is offline and dependency-free. ``popqa`` and ``triviaqa`` require
    the ``datasets`` package (only present on the GPU box) and hit the network.
    ``split=None`` (or empty) picks the per-dataset default labelled split.
    """
    name = name.lower()
    if name == "sample":
        return _sample_questions(limit)
    resolved = split or _DEFAULT_SPLIT.get(name, "test")
    if name == "popqa":
        return _load_popqa(limit, resolved, shuffle_seed)
    if name == "triviaqa":
        return _load_triviaqa(limit, resolved, shuffle_seed)
    raise ValueError(f"unknown dataset: {name!r} (use sample|popqa|triviaqa)")


def _hf_load(path, *args, **kwargs):
    try:
        from datasets import load_dataset as hf_load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - GPU-box path
        raise RuntimeError(
            "The 'datasets' package is required for popqa/triviaqa. "
            "Install GPU requirements (deploy/vast_setup.sh) or use dataset=sample."
        ) from exc
    return hf_load_dataset(path, *args, **kwargs)


def _maybe_shuffle(rows, shuffle_seed):  # pragma: no cover - GPU-box path
    if shuffle_seed is None:
        return rows
    return rows.shuffle(seed=shuffle_seed)


def _load_popqa(limit, split, shuffle_seed):  # pragma: no cover - GPU-box path
    ds = _hf_load("akariasai/PopQA", split=split)
    ds = _maybe_shuffle(ds, shuffle_seed)
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        possible = row.get("possible_answers")
        golds = _parse_possible_answers(possible, row.get("obj"))
        out.append(
            EvalQuestion(
                id=f"popqa-{row.get('id', i)}",
                question=row["question"],
                gold_answers=golds,
                metadata={
                    "popularity": row.get("s_pop") or row.get("o_pop"),
                    "relation": row.get("prop"),
                    "source": "popqa",
                },
            )
        )
    return out


def _parse_possible_answers(possible, obj):  # pragma: no cover - GPU-box path
    import json

    golds = []
    if isinstance(possible, str):
        try:
            golds = list(json.loads(possible))
        except (ValueError, TypeError):
            golds = [possible]
    elif isinstance(possible, (list, tuple)):
        golds = list(possible)
    if obj and obj not in golds:
        golds.append(obj)
    return [str(g) for g in golds if g]


def _load_triviaqa(limit, split, shuffle_seed):  # pragma: no cover - GPU-box path
    ds = _hf_load("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    ds = _maybe_shuffle(ds, shuffle_seed)
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        ans = row.get("answer", {}) or {}
        golds = []
        if ans.get("value"):
            golds.append(ans["value"])
        golds.extend(ans.get("aliases", []) or [])
        out.append(
            EvalQuestion(
                id=f"triviaqa-{row.get('question_id', i)}",
                question=row["question"],
                gold_answers=[str(g) for g in golds if g],
                metadata={"source": "triviaqa"},
            )
        )
    return out
