"""finetune.py with a chosen pad token, for models other than Llama (run_model_grid.sh ARM=padfix).

Same idea as diagnostics/finetune_padfix.py (stage 5, Llama), kept separate so that file
-- and every Llama result made with it -- stays untouched. finetune.py sets pad = eos
when the tokenizer has no pad; DataCollatorForLanguageModeling then masks every pad id
out of `labels`, so the token that ends each turn is never a training target and the
model never learns to stop. This runs finetune.main() unchanged except that, after
setup_model_and_tokenizer(), the pad token is replaced:

  --pad_token=<tok>   e.g. <unk> for OpenELM (Llama-2 tokenizer: id 0; byte-fallback
                      SentencePiece never emits it for real text, so it never occurs in
                      the data). The token must already be in the vocab with an embedding
                      row; it is looked up by exact string, so (unlike finetune_padfix.py)
                      the unk token itself is allowed.

Pad positions keep attention_mask 0 and label -100 either way; the pad id only decides
whether eos survives in `labels`. The patched tokenizer is what the Trainer saves with
each checkpoint, so evaluate.py / eval_generations.py / geometry.py (which load the
tokenizer from the checkpoint) pad with the same token. finetune.py is imported, never
modified. Usage (repo root):
  torchrun --nproc_per_node=4 diagnostics/finetune_pad.py [finetune.py args] --pad_token='<unk>'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pop_arg(name):
    """Remove --name=value / --name value from sys.argv (finetune's parser rejects it)."""
    for i, a in enumerate(sys.argv):
        if a.startswith(f"--{name}="):
            sys.argv.pop(i)
            return a.split("=", 1)[1]
        if a == f"--{name}":
            sys.argv.pop(i)
            return sys.argv.pop(i)
    return None


PAD_TOKEN = _pop_arg("pad_token")
if not PAD_TOKEN:
    sys.exit("finetune_pad.py: --pad_token is required")

import torch  # noqa: E402
import finetune  # noqa: E402 — repo-root finetune.py, imported, never modified

_setup_model_and_tokenizer = finetune.setup_model_and_tokenizer


def setup_model_and_tokenizer(*args, **kwargs):
    model, tokenizer = _setup_model_and_tokenizer(*args, **kwargs)
    vocab = tokenizer.get_vocab()
    assert PAD_TOKEN in vocab, f"{PAD_TOKEN!r} not in vocab"
    pad_id = vocab[PAD_TOKEN]
    assert pad_id < model.get_input_embeddings().weight.shape[0], f"{PAD_TOKEN!r} has no embedding row"
    assert pad_id != tokenizer.eos_token_id, f"{PAD_TOKEN!r} is the eos token"
    tokenizer.pad_token = PAD_TOKEN
    assert tokenizer.pad_token_id == pad_id
    model.config.pad_token_id = pad_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = pad_id
    if torch.cuda.current_device() == 0:
        print(f"[finetune_pad] pad={tokenizer.pad_token!r} ({tokenizer.pad_token_id})  "
              f"eos={tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
    return model, tokenizer


finetune.setup_model_and_tokenizer = setup_model_and_tokenizer

if __name__ == "__main__":
    finetune.main()
