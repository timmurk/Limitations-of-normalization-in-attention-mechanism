import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from attnnorm.metrics import sensitivity
from attnnorm.model import attention_tensors, load_gpt2
from attnnorm.plots import save
from attnnorm.text import read_text, token_chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--text", default=None)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--temps", default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    p.add_argument("--eps", default="1e-3,1e-1,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    temps = [float(x) for x in args.temps.split(",")]
    epss = [float(x) for x in args.eps.split(",")]

    model, tok = load_gpt2(args.model, args.device)
    ids = token_chunks(tok, read_text(args.text), args.length, 1, args.offset)[0]
    layers = attention_tensors(model, ids.to(args.device))
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    rows = []
    for temp in temps:
        for eps in epss:
            per_head = []
            for li, layer in enumerate(layers):
                g = sensitivity(layer["scores"].double(), temp, eps, gen).mean(-1)
                per_head += [(li, h, v) for h, v in enumerate(g.tolist())]
            li, h, v = max(per_head, key=lambda r: r[2])
            rows.append(dict(T=temp, eps=eps, max_g=v, mean_g=float(np.mean([r[2] for r in per_head])),
                             argmax_layer=li, argmax_head=h, bound=min(1 / (4 * temp), 2 ** 0.5)))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(args.out, "tables", "gradient.csv"), index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for eps, style in zip(epss, ["o-.", "^-", "s--"]):
        d = df[df.eps == eps]
        ax.plot(d["T"], d.max_g, style, lw=2, markersize=8, label=f"eps={eps:g}")
    b = df.drop_duplicates("T")
    ax.plot(b["T"], b.bound, "k:", lw=3, label=r"$\min\{1/(4T),\sqrt{2}\}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T")
    ax.set_ylabel(r"max over heads of $\|\Delta\alpha\|/\varepsilon$")
    ax.set_title("Gradient sensitivity")
    ax.legend()
    save(fig, os.path.join(args.out, "figures", "gradient.png"))


if __name__ == "__main__":
    main()
