"""Geometry metrics for LLM-JEPA checkpoints.

Frozen eval-mode forward passes over a data split; computes per-layer scalars:
  - rankme_{text,code}: effective rank (spectral entropy) of the embedding matrix
  - uniformity_{text,code}: Gaussian kernel potential on the unit hypersphere
    (Wang & Isola 2020, t=2)
  - alignment_cos: mean paired cosine similarity Text <-> Code
  - linear_r2_text_to_code: held-out R^2 of a ridge linear map Text -> Code

Writes tidy JSONL: one row per (lbd, k, seed, step, split, layer). Layers are
never averaged; pooling is over tokens within a single layer (last-token
readout, matching evaluate.py's similarity path).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from evaluate import (
    load_model_and_tokenizer,
    get_user_messages,
    get_assistant_messages,
    format_conversation,
)


def embed_all_layers(model, tokenizer, prompt, max_length):
    """Last-token embedding at every layer. Returns [num_layers+1, hidden_dim]."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of [1, seq_len, dim], index 0 is the embedding layer
    return torch.stack([hs[0, -1, :] for hs in outputs.hidden_states]).float().cpu()


def rankme(X):
    """Effective rank = exp(spectral entropy of singular values). X: [N, D]."""
    s = torch.linalg.svdvals(X.double())
    if s.sum() <= 0:
        return float("nan")
    p = s / s.sum() + 1e-12
    return torch.exp(-(p * torch.log(p)).sum()).item()


def uniformity(X, t=2.0):
    """Wang & Isola 2020 uniformity: log mean exp(-t * ||xi - xj||^2) on the sphere."""
    Xn = F.normalize(X, dim=-1)
    sq_pdist = torch.pdist(Xn, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().item()


def alignment_cos(X, Y):
    """Mean paired cosine similarity between matched rows of X and Y."""
    return F.cosine_similarity(X, Y, dim=-1).mean().item()


def linear_r2(X, Y, rel_alpha=1e-3, seed=0):
    """Held-out R^2 of a ridge linear map X -> Y (fit on half, score on half)."""
    X = X.double().numpy()
    Y = Y.double().numpy()
    n, d = X.shape
    perm = np.random.RandomState(seed).permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    x_mean, y_mean = X[tr].mean(0), Y[tr].mean(0)
    Xtr, Ytr = X[tr] - x_mean, Y[tr] - y_mean
    Xte, Yte = X[te] - x_mean, Y[te] - y_mean
    gram = Xtr.T @ Xtr
    trace = np.trace(gram)
    ss_tot = (Yte ** 2).sum()
    # Degenerate layer (e.g. layer 0 with last-token pooling: identical
    # template token for every example) — R^2 is undefined
    if trace <= 0 or ss_tot <= 0:
        return float("nan")
    alpha = max(rel_alpha * trace / d, 1e-10)
    W = np.linalg.solve(gram + alpha * np.eye(d), Xtr.T @ Ytr)
    ss_res = ((Yte - Xte @ W) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main():
    parser = argparse.ArgumentParser(description="Compute geometry metrics for a frozen checkpoint")
    parser.add_argument("--model_name", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--original_model_name", type=str, required=True, help="Base model name")
    parser.add_argument("--input_files", type=str, required=True,
                        help="Comma-separated split:path pairs, e.g. train:datasets/synth_train.jsonl,test:datasets/synth_test.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Tidy JSONL to append rows to")
    parser.add_argument("--max_examples", type=int, default=500, help="Examples sampled per split")
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed for subsampling examples")
    parser.add_argument("--max_length", type=int, default=512, help="Max tokenized sequence length")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="Device map for model loading")
    # Run metadata copied verbatim into every output row
    parser.add_argument("--lbd", type=float, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=str, required=True, help="Checkpoint step label")
    args = parser.parse_args()

    splits = []
    for pair in args.input_files.split(","):
        name, path = pair.split(":", 1)
        splits.append((name, path))

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.original_model_name, device_map=args.device_map
    )

    with open(args.output_file, "a") as out:
        for split_name, path in splits:
            dataset = load_dataset("json", data_files=path)["train"]
            if len(dataset) > args.max_examples:
                dataset = dataset.shuffle(seed=args.sample_seed).select(range(args.max_examples))
            print(f"[geometry] {args.model_name} split={split_name}: {len(dataset)} examples")

            text_embs, code_embs = [], []
            for example in tqdm(dataset, desc=f"embeddings ({split_name})"):
                messages = example["messages"]
                text_prompt = format_conversation(
                    get_user_messages(args.original_model_name, messages),
                    tokenizer, similarity=True)
                code_prompt = format_conversation(
                    get_assistant_messages(args.original_model_name, messages),
                    tokenizer, include_assistant=True, similarity=True)
                text_embs.append(embed_all_layers(model, tokenizer, text_prompt, args.max_length))
                code_embs.append(embed_all_layers(model, tokenizer, code_prompt, args.max_length))

            text_embs = torch.stack(text_embs)  # [N, L+1, D]
            code_embs = torch.stack(code_embs)

            for layer in range(text_embs.shape[1]):
                X = text_embs[:, layer, :]
                Y = code_embs[:, layer, :]
                row = {
                    "lbd": args.lbd,
                    "k": args.k,
                    "seed": args.seed,
                    "step": args.step,
                    "split": split_name,
                    "layer": layer,
                    "n": X.shape[0],
                    "rankme_text": rankme(X),
                    "rankme_code": rankme(Y),
                    "uniformity_text": uniformity(X),
                    "uniformity_code": uniformity(Y),
                    "alignment_cos": alignment_cos(X, Y),
                    "linear_r2_text_to_code": linear_r2(X, Y, seed=args.sample_seed),
                }
                out.write(json.dumps(row) + "\n")
            out.flush()

    print(f"[geometry] wrote rows to {args.output_file}")


if __name__ == "__main__":
    main()
