import torch


def sort_row(alpha, x):
    order = torch.argsort(alpha, descending=True)
    return alpha[order], x[order]


def context_distances(a, x, k):
    g = x @ x[:k].T
    p = torch.cumsum(g * a[None, :k], dim=1)
    s2 = torch.diagonal(torch.cumsum(a[:k, None] * p[:k], dim=0))
    sq = (a * a * (x * x).sum(1))[:, None] - 2 * a[:, None] * p + s2[None, :]
    return sq.clamp_min(0).sqrt(), g


def outside(n, k, device):
    return torch.ones(n, k, dtype=torch.bool, device=device).tril(-1)


def distance_curves(alpha, x, nmax=None):
    alpha, x = alpha.double(), x.double()
    n = len(alpha)
    k = n - 1 if nmax is None else min(nmax, n - 1)
    a, xs = sort_row(alpha, x)
    dist, g = context_distances(a, xs, k)
    out = outside(n, k, a.device)
    true = (dist * out).sum(0)

    big_n = torch.arange(1, k + 1, device=a.device, dtype=a.dtype)
    gu = x @ (x.T @ alpha)
    xbar2 = alpha @ gu
    norms2 = (x * x).sum(1)
    c = big_n / (n - 1)
    al = alpha[:, None]
    term = al ** 2 * (1 + c) ** 2 * norms2[:, None] - 2 * al * (1 + c) * c * gu[:, None] + c ** 2 * xbar2
    expected = (n - big_n) / n * term.clamp_min(0).sqrt().sum(0)

    norm = (xs * xs).sum(1).sqrt()
    pair = (norm[:, None] ** 2 + norm[None, :k] ** 2 - 2 * g).clamp_min(0).sqrt()
    pref = torch.cummax(pair, dim=1).values
    d1 = pref.masked_fill(~out, float("-inf")).max(0).values
    abar = torch.cumsum(a, 0)[:k]
    top_norm = torch.cummax(norm, 0).values[:k]
    bound = (1 - abar) * d1 + top_norm * (abar * (n - big_n) - (1 - abar))
    return true, expected, bound


def separability_curves(alpha, x, radius=1.0, nmax=None):
    alpha, x = alpha.double(), x.double()
    n = len(alpha)
    k = n - 1 if nmax is None else min(nmax, n - 1)
    xn = radius * x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gram = xn @ xn.T
    gram.fill_diagonal_(-float("inf"))
    delta2 = (2 * radius ** 2 - 2 * gram.max()).clamp_min(0)

    a, xs = sort_row(alpha, xn)
    dist, _ = context_distances(a, xs, k)
    inside = ~outside(n, k, a.device)
    r = dist.masked_fill(inside, float("inf")).min(0).values
    big_n = torch.arange(1, k + 1, device=a.device, dtype=a.dtype)
    ns = ((dist <= r[None, :]) & inside).sum(0).double()

    s1 = torch.cumsum(a, 0)[None, :k] - a[:, None]
    s2 = torch.cumsum(a * a, 0)[None, :k] - (a * a)[:, None]
    xi = (radius ** 2 * s2 + (radius ** 2 - delta2 / 2) * (s1 ** 2 - s2)).clamp_min(0).sqrt() * inside
    lower = 1 - xi.sum(0) / (r * big_n)
    upper = (torch.exp(-((r[None, :] - xi) ** 2) / (16 * radius ** 2)) * inside).sum(0) / big_n
    return ns / big_n, lower, upper


def sensitivity(scores, temp, eps, gen=None):
    valid = torch.isfinite(scores)
    v = torch.randn(scores.shape, generator=gen, device=scores.device, dtype=scores.dtype) * valid
    v = v / v.norm(dim=-1, keepdim=True)
    base = torch.softmax(scores / temp, -1)
    moved = torch.softmax((scores + eps * v) / temp, -1)
    return (moved - base).norm(dim=-1) / eps
