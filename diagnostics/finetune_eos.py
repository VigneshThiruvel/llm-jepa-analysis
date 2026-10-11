"""finetune.py with the tokenizer's eos appended to every full training conversation.

For models whose chat template ends a turn with its own marker but never writes the
tokenizer's eos (gemma-2: '<end_of_turn>\\n', eos = '<eos>'), the eos that
evaluate.py stops on (tokenizer.eos_token_id) never occurs in the training text, so
whether a fine-tuned model stops depends on what it does after the turn marker -- a
position that is never trained. This makes eos a training target and nothing else:

  after setup_model_and_tokenizer(), tokenizer.apply_chat_template is wrapped so that a
  call rendering a full conversation (add_generation_prompt=False, last message role
  'assistant', tokenize=False) gets tokenizer.eos_token appended.

Only finetune.py's full-conversation call matches. The JEPA Text view (role 'user') and
Code view (gemma: the assistant text recast as role 'user') are unchanged, so
--last_token indexing is unchanged; the prompt-length call (add_generation_prompt=True)
is unchanged; the eval prompts are unchanged (the wrapper is not saved with the
checkpoint's tokenizer). finetune.py is imported, never modified. Usage (repo root):
  torchrun --nproc_per_node=4 diagnostics/finetune_eos.py [finetune.py args]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import finetune  # noqa: E402 — repo-root finetune.py, imported, never modified

_setup_model_and_tokenizer = finetune.setup_model_and_tokenizer


def setup_model_and_tokenizer(*args, **kwargs):
    model, tokenizer = _setup_model_and_tokenizer(*args, **kwargs)
    assert tokenizer.eos_token_id is not None and tokenizer.eos_token_id != tokenizer.pad_token_id
    _apply = tokenizer.apply_chat_template

    def apply_chat_template(conversation, *a, **kw):
        out = _apply(conversation, *a, **kw)
        if (isinstance(out, str) and not kw.get("add_generation_prompt", False)
                and conversation and conversation[-1].get("role") == "assistant"):
            out = out + tokenizer.eos_token
        return out

    tokenizer.apply_chat_template = apply_chat_template
    if torch.cuda.current_device() == 0:
        print(f"[finetune_eos] appending {tokenizer.eos_token!r} ({tokenizer.eos_token_id}) to full conversations; "
              f"pad={tokenizer.pad_token!r} ({tokenizer.pad_token_id})")
    return model, tokenizer


finetune.setup_model_and_tokenizer = setup_model_and_tokenizer

if __name__ == "__main__":
    finetune.main()
