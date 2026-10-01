"""finetune.py with a pad token that is not the stop token (k=16 diagnosis, stage 5).

finetune.py sets pad = eos = <|eot_id|> (setup_model_and_tokenizer). The
DataCollatorForLanguageModeling it trains with rebuilds `labels` from input_ids
and sets every pad id to -100, so every <|eot_id|> -- including the one that ends
the answer -- is dropped from the loss: the model is never taught to stop. This
runs finetune.main() unchanged except for two monkeypatches:

  --pad_token=<tok>          pad with <tok> instead of eos. Default
                             <|finetune_right_pad_id|> (id 128004): Meta's reserved
                             fine-tuning pad token, already in the Llama-3.2 vocab
                             (no embedding resize, predictor ids unchanged), never
                             present in any dataset. --pad_token=eos keeps upstream
                             behaviour (the same-day control arm).
  --train_date_string=<d>    pin the date the Llama-3 chat template stamps into every
                             training prompt, so both arms train on identical text.

Pad positions carry attention_mask 0 and label -100 either way, so the pad id only
decides whether eot survives in `labels`. finetune.py itself is imported, never
modified. Usage (from the repo root, PYTHONPATH as diagnostics/common.sh):
  torchrun --nproc_per_node=4 diagnostics/finetune_padfix.py [finetune.py args] \
      --pad_token=<|finetune_right_pad_id|> --train_date_string="22 Sep 2026"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pop_arg(name, default):
    """Remove --name=value / --name value from sys.argv (finetune's parser rejects it)."""
    for i, a in enumerate(sys.argv):
        if a.startswith(f"--{name}="):
            sys.argv.pop(i)
            return a.split("=", 1)[1]
        if a == f"--{name}":
            sys.argv.pop(i)
            return sys.argv.pop(i)
    return default


PAD_TOKEN = _pop_arg("pad_token", "<|finetune_right_pad_id|>")
TRAIN_DATE = _pop_arg("train_date_string", None)

import torch  # noqa: E402
import finetune  # noqa: E402 — the repo-root finetune.py, imported, never modified

_setup_model_and_tokenizer = finetune.setup_model_and_tokenizer


def setup_model_and_tokenizer(*args, **kwargs):
    model, tokenizer = _setup_model_and_tokenizer(*args, **kwargs)
    if PAD_TOKEN != "eos":
        pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        assert pad_id is not None and pad_id != tokenizer.unk_token_id, f"{PAD_TOKEN} not in vocab"
        assert pad_id < model.get_input_embeddings().weight.shape[0], f"{PAD_TOKEN} has no embedding row"
        tokenizer.pad_token = PAD_TOKEN
        assert tokenizer.pad_token_id != tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    if TRAIN_DATE:
        _apply = tokenizer.apply_chat_template

        def apply_chat_template(*a, **kw):
            kw.setdefault("date_string", TRAIN_DATE)
            return _apply(*a, **kw)

        tokenizer.apply_chat_template = apply_chat_template
    if torch.cuda.current_device() == 0:
        print(f"[padfix] pad={tokenizer.pad_token!r} ({tokenizer.pad_token_id})  "
              f"eos={tokenizer.eos_token!r} ({tokenizer.eos_token_id})  train_date={TRAIN_DATE!r}")
    return model, tokenizer


finetune.setup_model_and_tokenizer = setup_model_and_tokenizer

if __name__ == "__main__":
    finetune.main()
