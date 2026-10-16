import sys

import torch
from transformers import AutoModelForCausalLM

from attnnorm.metrics import distance_curves, separability_curves
from attnnorm.model import attention_tensors


def direct(alpha, x, radius):
    n = len(alpha)
    order = torch.argsort(alpha, descending=True)
    out = {k: [] for k in ["true", "expected", "bound", "ratio", "lower", "upper"]}
    xbar = (alpha[:, None] * x).sum(0)
    xn = radius * x / x.norm(dim=1, keepdim=True)
    delta = min((xn[i] - xn[j]).norm() for i in range(n) for j in range(n) if i != j)
    for N in range(1, n):
        sel, rest = order[:N], order[N:]
        s = (alpha[sel, None] * x[sel]).sum(0)
        out["true"].append(sum((alpha[i] * x[i] - s).norm() for i in rest))
        c = N / (n - 1)
        out["expected"].append((n - N) / n * sum((alpha[i] * (1 + c) * x[i] - c * xbar).norm() for i in range(n)))
        abar = alpha[sel].sum()
        d1 = max((x[i] - x[j]).norm() for i in rest for j in sel)
        top = max(x[j].norm() for j in sel)
        out["bound"].append((1 - abar) * d1 + top * (abar * (n - N) - (1 - abar)))
        sn = (alpha[sel, None] * xn[sel]).sum(0)
        r = min((alpha[i] * xn[i] - sn).norm() for i in rest)
        ns = sum(int((alpha[i] * xn[i] - sn).norm() <= r) for i in sel)
        out["ratio"].append(ns / N)
        xis = []
        for i in sel:
            others = [j for j in sel if j != i]
            s2 = sum(alpha[j] ** 2 for j in others)
            cross = sum(alpha[j] * alpha[k] for j in others for k in others if j != k)
            xis.append(max(radius ** 2 * s2 + (radius ** 2 - delta ** 2 / 2) * cross, 0) ** 0.5)
        out["lower"].append(1 - sum(xis) / (r * N))
        out["upper"].append(sum(torch.exp(-((r - xi) ** 2) / (16 * radius ** 2)) for xi in xis) / N)
    return {k: torch.tensor([float(v) for v in vals], dtype=torch.float64) for k, vals in out.items()}


def check_formulas():
    torch.manual_seed(0)
    for n in [2, 3, 7, 12]:
        alpha = torch.softmax(torch.randn(n, dtype=torch.float64) * 2, 0)
        x = torch.randn(n, 5, dtype=torch.float64) * 3
        ref = direct(alpha, x, 1.5)
        got = dict(zip(["true", "expected", "bound"], distance_curves(alpha, x)))
        got.update(zip(["ratio", "lower", "upper"], separability_curves(alpha, x, 1.5)))
        for k in ref:
            assert torch.allclose(got[k], ref[k], rtol=1e-9, atol=1e-9), (n, k, got[k], ref[k])
    print("formulas ok")


def check_model(path):
    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
    ids = torch.randint(0, model.config.vocab_size, (40,))
    layers = attention_tensors(model, ids)
    ref = model(input_ids=ids[None], output_attentions=True).attentions
    for mine, theirs in zip(layers, ref):
        assert torch.allclose(mine["attn"], theirs[0], atol=1e-6)
    print(path, "attention ok")


if __name__ == "__main__":
    check_formulas()
    for path in sys.argv[1:]:
        check_model(path)
