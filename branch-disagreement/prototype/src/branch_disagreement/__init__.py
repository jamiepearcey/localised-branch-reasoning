"""branch-disagreement: cheap hallucination detection via prefix-shared branches.

Pure-logic modules (config, normalize, datasets, scoring, metrics, the proxy
runner) import with the standard library only. Heavy dependencies (torch, vllm,
transformers, numpy) live behind the vLLM runner and the NLI clusterer and are
imported lazily, so the test suite and the CPU proxy run need no GPU stack.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
