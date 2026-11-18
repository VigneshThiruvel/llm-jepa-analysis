"""finetune.py + padfix with a JEPA term that cannot be solved by collapse (Phase C).

Phase A (LOGBOOK 2026-10-23) found that the cosine term is satisfied by collapse: every
Text and every Code at the pooled position maps to ~one vector (pos cos = neg cos =
0.999, retrieval at chance, rank 2-4). This wrapper replaces only the JEPA term of
RepresentationTrainer.compute_loss. The LM loss, the additive-mask packing, the data and
the optimizer are finetune.py's own, and the pad fix + date pin come from
finetune_padfix (imported). Extra flags, popped before finetune's parser sees argv:

  --jepa_obj=cos|infonce|vicreg   cos      1 - mean cos(T_i, C_i)  (finetune.py's default term)
                                  infonce  cross-entropy of T_i against all C in the batch
                                           gathered over ranks (64-way at 4 GPUs x bs 16),
                                           temperature --jepa_tau (0.07, as finetune.py)
                                  vicreg   1 - cos + --vic_var * mean relu(1 - std_d)
                                           + --vic_cov * sum offdiag(cov)^2 / D, per side,
                                           on the gathered batch
  --jepa_layer=<int>              hidden_states index to read (-1 = post-norm last layer,
                                  as finetune.py; 8 = output of block 8)
  --jepa_pool=last|mean           last  finetune.py's positions (--last_token, packed)
                                  mean  mean over each branch's content span: user message
                                        (+ predictor markers) / answer tokens, i.e. from 2
                                        tokens after the branch's last <|end_header_id|>
                                        through the last_token position
  --jepa_gather=1|0               gather embeddings across ranks for infonce/vicreg
  --jepa_obj=infonce_proj         infonce on g(T), g(C), g = Linear(D,D)-GELU-Linear(D,256), a
                                  trainable head on the model (stripped from every save)
  --jepa_obj=byol                 2 - 2 cos(q(g(T)), sg(g_t(C_t))), one-sided Text->Code; q =
                                  Linear(256,512)-GELU-Linear(512,256); g_t / C_t from an EMA
                                  copy of the whole model (--byol_tau 0.996), updated at the start
                                  of every step (grad_accum must be 1), second no-grad forward

  --jepa_obj=cos --jepa_layer=-1 --jepa_pool=last reproduces finetune.py's term exactly.
Requires --additive_mask (the only JEPA path the sweep uses). Logs, besides lm_loss /
jepa_loss: jepa_pos (mean pos cos), jepa_gap (pos - mean neg cos) and jepa_r1 (top-1
T->C retrieval within the gathered batch), all detached diagnostics.

  torchrun --nproc_per_node=4 diagnostics/finetune_jepa_obj.py [finetune.py args] \
      --additive_mask --batch_size=16 --grad_accum=1 --jepa_obj=infonce \
      --pad_token=<|finetune_right_pad_id|> --train_date_string="22 Sep 2026"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import finetune_padfix  # noqa: E402 — pops --pad_token/--train_date_string, patches setup
from finetune_padfix import _pop_arg  # noqa: E402

JEPA_OBJ = _pop_arg("jepa_obj", "cos")
JEPA_TAU = float(_pop_arg("jepa_tau", "0.07"))
JEPA_LAYER = int(_pop_arg("jepa_layer", "-1"))
JEPA_POOL = _pop_arg("jepa_pool", "last")
JEPA_GATHER = _pop_arg("jepa_gather", "1") == "1"
VIC_VAR = float(_pop_arg("vic_var", "1.0"))
VIC_COV = float(_pop_arg("vic_cov", "0.04"))
BYOL_TAU = float(_pop_arg("byol_tau", "0.996"))
PROJ_DIM = 256
assert JEPA_OBJ in ("cos", "infonce", "vicreg", "infonce_proj", "byol"), JEPA_OBJ
assert JEPA_POOL in ("last", "mean"), JEPA_POOL

import copy  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.distributed.nn.functional import all_gather  # noqa: E402

import finetune  # noqa: E402 — imported, never modified

HEADS = ("jepa_head", "jepa_pred")  # trainable modules hung on the model, never saved
END_HEADER = 128007  # <|end_header_id|>; content starts after it and "\n\n" (271)


def gather(x):
    if not (JEPA_GATHER and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return x, 0
    return torch.cat(all_gather(x), 0), dist.get_rank() * x.shape[0]


def content_start(row, end):
    """Index of the first content token of the branch whose last token is at `end`."""
    hdr = (row[:end + 1] == END_HEADER).nonzero().flatten()
    start = int(hdr[-1]) + 2 if len(hdr) else int(end)
    return min(start, int(end))


def pool(hidden, packed_ids, ends):
    """[B, D] embeddings at `ends` (last) or averaged over each branch's content span (mean)."""
    r = torch.arange(hidden.shape[0], device=hidden.device)
    if JEPA_POOL == "last":
        return hidden[r, ends]
    out = []
    for i in range(hidden.shape[0]):
        e = int(ends[i])
        s = content_start(packed_ids[i], e)
        out.append(hidden[i, s:e + 1].mean(0))
    return torch.stack(out)


def mlp(d_in, d_hid, d_out, like):
    w = like.get_input_embeddings().weight
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out)).to(w.device, w.dtype)


def run(m, x):
    """Apply a head in its own dtype (hidden states and heads can differ in dtype)."""
    return m(x.to(next(m.parameters()).dtype))


_setup = finetune.setup_model_and_tokenizer  # finetune_padfix's wrapper


def setup_model_and_tokenizer(*args, **kwargs):
    model, tokenizer = _setup(*args, **kwargs)
    if JEPA_OBJ in ("infonce_proj", "byol"):
        D = model.config.hidden_size
        torch.manual_seed(kwargs.get("seed") or 0)
        model.jepa_head = mlp(D, D, PROJ_DIM, model)
        if JEPA_OBJ == "byol":
            model.jepa_pred = mlp(PROJ_DIM, 2 * PROJ_DIM, PROJ_DIM, model)
    return model, tokenizer


finetune.setup_model_and_tokenizer = setup_model_and_tokenizer

_save = finetune.RepresentationTrainer._save


def _save_without_heads(self, *args, **kwargs):
    m = self.accelerator.unwrap_model(self.model)
    held = {h: m._modules.pop(h) for h in HEADS if h in m._modules}
    try:
        return _save(self, *args, **kwargs)
    finally:
        m._modules.update(held)


_build = finetune.RepresentationTrainer.build_with_additive_mask


def build_with_additive_mask(self, inputs):
    out = _build(self, inputs)
    self._llm_inputs = out[0]  # kept for byol's target forward
    return out


@torch.no_grad()
def byol_target(self, model):
    """EMA target, created from the online model on the first call and updated on every later one."""
    online = self.accelerator.unwrap_model(model)
    if getattr(self, "_target", None) is None:
        self._target = copy.deepcopy(online).eval()
        self._target.requires_grad_(False)
    else:
        for t, o in zip(self._target.parameters(), online.parameters()):
            t.lerp_(o.detach(), 1.0 - BYOL_TAU)
    return self._target


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    assert self.additive_mask, "finetune_jepa_obj.py supports the --additive_mask path only"
    fr = self.forward(model, inputs)  # finetune's forward: packs Text+Code, one model call
    main_outputs = fr["main_outputs"]
    lm_loss = main_outputs.loss
    if fr["user_hidden_states"] is None:  # jepa_ratio skip
        jepa_loss = 0.0
    else:
        B = inputs["input_ids"].shape[0]
        hs = main_outputs.hidden_states[JEPA_LAYER][B:2 * B]  # the packed Text|Code rows
        packed = inputs["input_ids_user"]  # build_with_additive_mask packed Code in, in place
        u = pool(hs, packed, self._last_token_user)
        a = pool(hs, packed, self._last_token_assistant)
        if os.environ.get("JEPA_OBJ_DEBUG") and not getattr(self, "_dbg_done", False) \
                and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            self._dbg_done = True
            tok = self.processing_class
            for name, ends in (("text", self._last_token_user), ("code", self._last_token_assistant)):
                e = int(ends[0])
                s0 = content_start(packed[0], e) if JEPA_POOL == "mean" else e
                print(f"[jepa_obj debug] {name} span [{s0},{e}] layer {JEPA_LAYER}: "
                      f"{tok.decode(packed[0, s0:e + 1])!r}  next={tok.decode(packed[0, e + 1:e + 2])!r}", flush=True)
        head = self.accelerator.unwrap_model(model)._modules.get("jepa_head")
        if JEPA_OBJ == "infonce_proj":
            u, a = run(head, u), run(head, a)  # gate stats below are on the projected z; the probe reads raw h
        pos = F.cosine_similarity(u, a, dim=-1)
        if JEPA_OBJ == "byol":
            assert self.args.gradient_accumulation_steps == 1, "byol's EMA step assumes grad_accum 1"
            tgt = byol_target(self, model)
            li = self._llm_inputs
            with torch.no_grad():
                ht = tgt(input_ids=li["input_ids"][B:2 * B], attention_mask=li["attention_mask"][B:2 * B],
                         output_hidden_states=True).hidden_states[JEPA_LAYER]
                zt = run(tgt.jepa_head, pool(ht, packed, self._last_token_assistant))
            zu = run(head, u)
            p = run(self.accelerator.unwrap_model(model).jepa_pred, zu)
            jepa_loss = 2.0 - 2.0 * F.cosine_similarity(p.float(), zt.float(), dim=-1).mean()
            self._byol_tgt_pos = float(F.cosine_similarity(zu.detach().float(), zt.float(), dim=-1).mean())
        elif JEPA_OBJ == "cos":
            jepa_loss = 1.0 - pos.mean()
        elif JEPA_OBJ in ("infonce", "infonce_proj"):
            a_all, off = gather(a)
            logits = F.normalize(u, dim=-1) @ F.normalize(a_all, dim=-1).T / JEPA_TAU
            jepa_loss = F.cross_entropy(logits, torch.arange(B, device=u.device) + off)
        else:
            u_all, _ = gather(u)
            a_all, _ = gather(a)
            reg = 0.0
            for z in (u_all.float(), a_all.float()):
                z = z - z.mean(0)
                std = torch.sqrt(z.var(0) + 1e-4)
                cov = (z.T @ z) / (z.shape[0] - 1)
                off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
                reg = reg + VIC_VAR * F.relu(1 - std).mean() + VIC_COV * off_diag / z.shape[1]
            jepa_loss = 1.0 - pos.mean() + reg
        with torch.no_grad():  # diagnostics on the gathered batch
            ua, _ = gather(u.detach())
            aa, _ = gather(a.detach())
            S = F.normalize(ua.float(), dim=-1) @ F.normalize(aa.float(), dim=-1).T
            n = S.shape[0]
            self._diag = {
                "jepa_pos": float(S.diagonal().mean()),
                "jepa_gap": float(S.diagonal().mean() - (S.sum() - S.diagonal().sum()) / max(n * (n - 1), 1)),
                "jepa_r1": float((S.argmax(1) == torch.arange(n, device=S.device)).float().mean()),
                "jepa_n": n,
            }
            if JEPA_OBJ == "byol":
                self._diag["byol_tgt_pos"] = self._byol_tgt_pos
    total_loss = self.gamma * lm_loss + self.lbd * jepa_loss
    self._last_lm_loss = float(lm_loss.detach())
    self._last_jepa_loss = float(jepa_loss.detach()) if torch.is_tensor(jepa_loss) else float(jepa_loss)
    return (total_loss, main_outputs) if return_outputs else total_loss


_log = finetune.RepresentationTrainer.log


def log(self, logs, *args, **kwargs):
    if "loss" in logs:
        for k, v in getattr(self, "_diag", {}).items():
            logs.setdefault(k, v)
    return _log(self, logs, *args, **kwargs)


finetune.RepresentationTrainer.compute_loss = compute_loss
finetune.RepresentationTrainer.log = log
finetune.RepresentationTrainer._save = _save_without_heads
finetune.RepresentationTrainer.build_with_additive_mask = build_with_additive_mask

if __name__ == "__main__":
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[jepa_obj] obj={JEPA_OBJ} tau={JEPA_TAU} layer={JEPA_LAYER} pool={JEPA_POOL} "
              f"gather={JEPA_GATHER} vic_var={VIC_VAR} vic_cov={VIC_COV} byol_tau={BYOL_TAU}", flush=True)
    finetune.main()
