"""Semantic clustering of branch answers.

A cluster is a group of branch answers that mean the same thing. Two clusterers:

- ``ExactMatchClusterer``: dependency-free, groups by normalized string equality.
  Used by the proxy path and as a fallback when no NLI model is available.
- ``NLIClusterer``: bidirectional-entailment clustering (Kuhn et al, 2023) using
  a DeBERTa-MNLI model. Imported lazily; only available on the GPU box.

Both return ``List[List[int]]`` — clusters of indices into the answers list,
ordered largest-first.
"""

from typing import List, Sequence

from .normalize import normalize_answer


def _order_clusters(clusters: List[List[int]]) -> List[List[int]]:
    clusters.sort(key=lambda c: (-len(c), min(c)))
    return clusters


class ExactMatchClusterer:
    name = "exact_match"

    def cluster(self, answers: Sequence[str]) -> List[List[int]]:
        groups: dict = {}
        for i, ans in enumerate(answers):
            key = normalize_answer(ans)
            groups.setdefault(key, []).append(i)
        return _order_clusters(list(groups.values()))


class NLIClusterer:  # pragma: no cover - GPU-box path
    """Cluster by bidirectional entailment.

    Two answers join the same cluster if the model entails them in both
    directions when each is framed in the question's context. Greedy
    single-link clustering against existing cluster representatives.
    """

    name = "nli_entailment"

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: str = "cuda"):
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        # DeBERTa-MNLI label order is [contradiction, neutral, entailment].
        self.entail_idx = 2

    def _entails(self, premise: str, hypothesis: str) -> bool:
        import torch

        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        pred = int(torch.argmax(logits).item())
        return pred == self.entail_idx

    def cluster(self, answers: Sequence[str], context: str = "") -> List[List[int]]:
        clusters: List[List[int]] = []
        reps: List[int] = []  # representative index per cluster
        prefix = f"{context} " if context else ""
        for i, ans in enumerate(answers):
            placed = False
            for c, rep in enumerate(reps):
                a = f"{prefix}{answers[rep]}"
                b = f"{prefix}{ans}"
                if self._entails(a, b) and self._entails(b, a):
                    clusters[c].append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
                reps.append(i)
        return _order_clusters(clusters)


def get_clusterer(engine: str, nli_model: str, device: str = "cuda"):
    """Pick a clusterer. NLI on the GPU engine, exact match otherwise.

    Falls back to exact match if the NLI stack cannot be loaded, so a missing
    model never aborts a run — it degrades to the lexical clusterer with a note.
    """
    if engine == "vllm":
        try:  # pragma: no cover - GPU-box path
            return NLIClusterer(nli_model, device=device)
        except Exception as exc:  # pragma: no cover - GPU-box path
            print(f"[clustering] NLI unavailable ({exc}); falling back to exact match")
            return ExactMatchClusterer()
    return ExactMatchClusterer()
