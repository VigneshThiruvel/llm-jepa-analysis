"""OpenELM generation shim, auto-imported (sitecustomize) when this directory is on PYTHONPATH.

apple/OpenELM-1_1B-Instruct's remote modeling code (2024, transformers 4.39 era)
reads `DynamicCache.seen_tokens` and `DynamicCache.get_max_length()` in
prepare_inputs_for_generation. transformers 4.55.2 (the authors' pinned stack,
setup.sh) has removed both, so every model.generate() call -- evaluate.py,
diagnostics/eval_generations.py -- dies with AttributeError on the first step.

This restores the two accessors with their old meaning, and nothing else:
  seen_tokens      -> get_seq_length()  (tokens already in the cache; for a
                      DynamicCache the two were always equal)
  get_max_length() -> None              (a DynamicCache has no maximum length)
Each is added only if missing, so on a transformers that still has them this is a
no-op. Training and the geometry forward passes never touch the cache. Only
diagnostics/run_model_grid.sh puts this directory on PYTHONPATH (MODEL_KEY=openelm);
the Llama sweep never sees it.
"""

try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None
except Exception:  # never break interpreter start-up over a shim
    pass
