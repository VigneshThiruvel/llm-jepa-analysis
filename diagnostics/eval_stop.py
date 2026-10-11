"""Run evaluate.py or diagnostics/eval_generations.py with a different stop token.

Both scripts stop generation on tokenizer.eos_token_id (evaluate.generate_response
passes it explicitly, overriding the model's generation_config). For gemma-2 that is
'<eos>' although the chat template ends the turn with '<end_of_turn>'. This wrapper
patches evaluate.load_model_and_tokenizer so the loaded tokenizer's eos_token is
--stop_token, then runs the target unchanged; prompts are identical as long as the chat
template does not reference eos_token (gemma's does not — checked by the assert).
Decoding still uses skip_special_tokens=True, so the stop token never reaches the
scored string. eval_generations' "eos" statistics (stop reason, p(eos) after gold) then
refer to --stop_token. Neither script is modified. Usage (repo root):
  python3 diagnostics/eval_stop.py --stop_token='<end_of_turn>' evaluate.py [evaluate.py args]
  python3 diagnostics/eval_stop.py --stop_token='<end_of_turn>' diagnostics/eval_generations.py [args]
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STOP = None
for i, a in enumerate(sys.argv):
    if a.startswith("--stop_token="):
        STOP = sys.argv.pop(i).split("=", 1)[1]
        break
if not STOP or len(sys.argv) < 2:
    sys.exit(__doc__)
target = sys.argv.pop(1)

import evaluate  # noqa: E402 — repo-root evaluate.py, imported, never modified

_load = evaluate.load_model_and_tokenizer


def load_model_and_tokenizer(*args, **kwargs):
    model, tokenizer = _load(*args, **kwargs)
    assert STOP in tokenizer.get_vocab(), f"{STOP!r} not in vocab"
    assert "eos_token" not in (tokenizer.chat_template or ""), "template uses eos_token; prompts would change"
    old = (tokenizer.eos_token, tokenizer.eos_token_id)
    pad = tokenizer.pad_token
    tokenizer.eos_token = STOP
    tokenizer.pad_token = pad  # unchanged; set explicitly in case pad was aliased to eos
    print(f"[eval_stop] stop token {old[0]!r} ({old[1]}) -> {tokenizer.eos_token!r} ({tokenizer.eos_token_id}); "
          f"pad {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")
    return model, tokenizer


evaluate.load_model_and_tokenizer = load_model_and_tokenizer

if os.path.basename(target) == "evaluate.py":
    sys.argv[0] = target
    evaluate.main()
else:
    sys.argv[0] = target
    runpy.run_path(target, run_name="__main__")
