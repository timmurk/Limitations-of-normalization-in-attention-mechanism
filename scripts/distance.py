import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attnnorm.plots import grouped_boxes, save


def select(data, metric, L, N):
    li = list(data["l_values"]).index(L)
    ni = list(data["n_values"]).index(N)
    return data[metric][..., li, ni].ravel()


def panel(data, pairs, labels, title, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    series = {m: [select(data, m, L, N) for L, N in pairs] for m in ["expected", "true", "bound"]}
    grouped_boxes(ax, labels, series)
    ax.set_yscale("log")
    ax.set_ylabel("Distance")
    ax.set_title(title)
    ax.legend()
    save(fig, path)
    rows = []
    for (L, N), lab in zip(pairs, labels):
        row = dict(L=L, N=N)
        for m in series:
            v = select(data, m, L, N)
            row[f"{m}_median"] = np.nanmedian(v)
            row[f"{m}_mean"] = np.nanmean(v)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/collect_gpt2.npz")
    p.add_argument("--n-fixed", type=int, default=5)
    p.add_argument("--l-fixed", type=int, default=1024)
    p.add_argument("--l-list", default="32,64,128,256,512,1024")
    p.add_argument("--n-list", default="1,5,10,20,100")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    data = np.load(args.data)
    ls = [int(x) for x in args.l_list.split(",") if int(x) in data["l_values"]]
    ns = [int(x) for x in args.n_list.split(",") if int(x) in data["n_values"]]

    t1 = panel(data, [(L, args.n_fixed) for L in ls], [f"L={L}" for L in ls],
               f"Distances for different L (N={args.n_fixed})", os.path.join(args.out, "figures", "distance_vs_L.png"))
    t2 = panel(data, [(args.l_fixed, N) for N in ns], [f"N={N}" for N in ns],
               f"Distances for different top-N (L={args.l_fixed})", os.path.join(args.out, "figures", "distance_vs_N.png"))
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    t1.to_csv(os.path.join(args.out, "tables", "distance_vs_L.csv"), index=False)
    t2.to_csv(os.path.join(args.out, "tables", "distance_vs_N.csv"), index=False)
    print(t1.to_string(index=False))
    print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
