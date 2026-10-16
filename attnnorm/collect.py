import numpy as np
import torch

from .metrics import distance_curves, separability_curves

METRICS = ["true", "expected", "bound", "ns_ratio", "ns_lower", "ns_upper"]


def head_rows(alpha, x, n_values, radius):
    L = alpha.shape[0]
    rows = {m: np.full((L, len(n_values)), np.nan) for m in METRICS}
    idx = torch.tensor(n_values) - 1
    nmax = max(n_values)
    for t in range(1, L):
        a, xs = alpha[t, : t + 1], x[: t + 1]
        ok = idx < t
        if not ok.any():
            continue
        sel = idx[ok]
        curves = distance_curves(a, xs, nmax) + separability_curves(a, xs, radius, nmax)
        for m, c in zip(METRICS, curves):
            rows[m][t, ok.numpy()] = c[sel].cpu().numpy()
    return rows


def by_length(rows, l_values):
    out = {}
    for m, r in rows.items():
        valid = ~np.isnan(r)
        csum = np.cumsum(np.where(valid, r, 0.0), 0)
        ccount = np.cumsum(valid, 0)
        picks = [L - 1 for L in l_values]
        with np.errstate(invalid="ignore"):
            out[m] = csum[picks] / ccount[picks]
    return out


def collect(layers, l_values, n_values, vectors="value", radius=1.0):
    n_layers = len(layers)
    n_heads = layers[0]["attn"].shape[0]
    result = {m: np.full((n_layers, n_heads, len(l_values), len(n_values)), np.nan) for m in METRICS}
    for li, layer in enumerate(layers):
        for h in range(n_heads):
            x = layer["value"][h] if vectors == "value" else layer["input"]
            rows = head_rows(layer["attn"][h], x, n_values, radius)
            for m, arr in by_length(rows, l_values).items():
                result[m][li, h] = arr
        print("layer", li, "done", flush=True)
    return result
