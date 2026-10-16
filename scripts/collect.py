import argparse
import os

import numpy as np
import torch

from attnnorm.collect import METRICS, collect
from attnnorm.model import attention_tensors, load_gpt2
from attnnorm.text import read_text, token_chunks

L_VALUES = [32, 64, 128, 256, 512, 1024] + list(range(50, 1001, 50))
N_VALUES = list(range(1, 97)) + [100, 128, 256]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--text", default=None)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--vectors", default="value", choices=["value", "input"])
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--l-values", default=None)
    p.add_argument("--n-values", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs/collect_gpt2.npz")
    args = p.parse_args()

    l_values = sorted({int(x) for x in args.l_values.split(",")} if args.l_values else set(L_VALUES))
    l_values = [L for L in l_values if L <= args.length]
    n_values = sorted({int(x) for x in args.n_values.split(",")} if args.n_values else set(N_VALUES))

    model, tok = load_gpt2(args.model, args.device)
    chunks = token_chunks(tok, read_text(args.text), args.length, args.chunks, args.offset)
    runs = {m: [] for m in METRICS}
    for c, ids in enumerate(chunks):
        layers = attention_tensors(model, ids.to(args.device))
        res = collect(layers, l_values, n_values, args.vectors, args.radius)
        for m in METRICS:
            runs[m].append(res[m])
        print("chunk", c, "done", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, l_values=np.array(l_values), n_values=np.array(n_values),
                        model=args.model, vectors=args.vectors,
                        **{m: np.stack(v) for m, v in runs.items()})
    print("saved", args.out)


if __name__ == "__main__":
    main()
