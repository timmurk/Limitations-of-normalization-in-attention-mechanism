import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from attnnorm.plots import save


def critical_n(data, L, alpha):
    li = list(data["l_values"]).index(L)
    for ni, N in enumerate(data["n_values"]):
        if N >= L:
            break
        a = data["true"][..., li, ni].ravel()
        b = data["expected"][..., li, ni].ravel()
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        p = ks_2samp(a, b).pvalue
        if p > alpha:
            return int(N), p
    return -1, np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/collect_gpt2.npz")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--l-list", default=",".join(str(x) for x in range(50, 1001, 50)))
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    data = np.load(args.data)
    ls = [int(x) for x in args.l_list.split(",") if int(x) in data["l_values"]]
    rows = []
    for L in ls:
        n, pv = critical_n(data, L, args.alpha)
        rows.append(dict(L=L, critical_n=n, p_value=pv, fraction=n / L if n > 0 else np.nan))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(args.out, "tables", "critical_n.csv"), index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ok = df.critical_n > 0
    ax.plot(df.L[ok], df.critical_n[ok], "o-", color="#8b0000", lw=2.5, label="Critical N")
    ax.set_xlabel("L")
    ax.set_ylabel("N")
    ax.set_title(f"Critical top-N (KS test, alpha={args.alpha})")
    ax.legend()
    save(fig, os.path.join(args.out, "figures", "critical_n.png"))


if __name__ == "__main__":
    main()
