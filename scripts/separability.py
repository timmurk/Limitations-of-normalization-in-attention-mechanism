import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attnnorm.plots import grouped_boxes, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/collect_gpt2.npz")
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--n-list", default="1,2,4,8,16,32,64,128,256")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    data = np.load(args.data)
    li = list(data["l_values"]).index(args.length)
    ns = [int(x) for x in args.n_list.split(",") if int(x) in data["n_values"]]
    idx = [list(data["n_values"]).index(N) for N in ns]
    series = {m: [data[m][..., li, i].ravel() for i in idx] for m in ["ns_ratio", "ns_lower", "ns_upper"]}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped_boxes(ax, [f"N={N}" for N in ns], series)
    ax.set_ylim(max(ax.get_ylim()[0], -1.0), 1.05)
    ax.set_ylabel(r"$N_s/N$")
    ax.set_title(f"Geometric separability (L={args.length})")
    ax.legend(loc="lower left")
    save(fig, os.path.join(args.out, "figures", "separability.png"))

    rows = []
    for k, N in enumerate(ns):
        row = dict(L=args.length, N=N)
        for m, vals in series.items():
            v = vals[k]
            row[f"{m}_median"] = np.nanmedian(v)
            row[f"{m}_q25"] = np.nanpercentile(v, 25)
            row[f"{m}_q75"] = np.nanpercentile(v, 75)
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(args.out, "tables", "separability.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
