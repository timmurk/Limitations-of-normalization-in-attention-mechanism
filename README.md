# Limitations of Normalization in Attention Mechanism

Code for

**Limitations of Normalization in Attention Mechanism**
Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, Radu State
[arXiv:2508.17821](https://arxiv.org/abs/2508.17821)

The scripts reproduce the GPT-2 experiments of Section 3: the distance bounds of Theorem 1,
the critical top-N obtained with a Kolmogorov–Smirnov test, the geometric separability of Theorem 2,
and the gradient sensitivity of softmax attention (Corollary 3).
Everything runs on a CPU.

## Layout

```
attnnorm/
  model.py      GPT-2 loading, per-head attention scores, weights, value vectors and layer inputs
  text.py       War and Peace from Project Gutenberg (or any local text file), token chunks
  metrics.py    true distance, expectation and upper bound (Th. 1), N_s/N and its bounds (Th. 2),
                finite-difference sensitivity of softmax
  collect.py    per-head statistics for every query row, averaged for each context length L
  plots.py      box-plot helpers
scripts/
  collect.py        runs GPT-2 once and stores all statistics in an .npz file
  distance.py       Fig. 2a, 2b: distances vs L (N = 5) and vs N (L = 1024)
  critical_n.py     Fig. 2c: smallest N for which true and expected distances pass the KS test
  separability.py   Fig. 3: N_s/N with the lower and upper bounds of Theorem 2
  gradient.py       Fig. 4: max over heads of ||softmax((l + eps v)/T) - softmax(l/T)|| / eps
run/               shell entry points
tests/             checks the vectorised formulas against direct loops and the attention against Hugging Face
results/paper/     figures from the paper
```

## Install

```
pip install -r requirements.txt
```

The first run downloads *War and Peace* (Project Gutenberg #2600, public domain) to `data/`
and starts the text at its opening line. `TEXT=path/to/file.txt` uses any other text instead.

## Running

```
bash run/smoke.sh          # tests + a 128-token run of every script
bash run/all.sh            # all experiments for GPT-2 (124M), L = 1024
bash run/models.sh         # the same for gpt2, gpt2-medium, gpt2-large
```

or step by step:

```
bash run/collect.sh        # outputs/collect_gpt2.npz
bash run/distance.sh       # distance_vs_L, distance_vs_N, critical_n
bash run/separability.sh   # separability
bash run/gradient.sh       # gradient
```

Figures go to `outputs/<model>/figures`, tables to `outputs/<model>/tables`.
Environment variables: `MODEL`, `TEXT`, `LENGTH`, `CHUNKS`, `VECTORS`, `ALPHA`, `OUT`, `DEVICE`.

`collect.py` takes about 16 s per head on one CPU core at L = 1024, i.e. roughly 40 minutes for the
144 heads of GPT-2; PyTorch uses all available cores by default.

## What is computed

For every layer, head and query row `t`, the row's attention weights over keys `0..t` are sorted, the top-N
set `I_N` is taken, and

* **true distance** `sum_{i not in I_N} ||alpha_i x_i - s||`, `s = sum_{i in I_N} alpha_i x_i` (Eq. 7);
* **expectation** over a uniformly random `I_N`, `(n-N)/n * sum_i ||alpha_i (1+c) x_i - c x_bar||`, `c = N/(n-1)`,
  `x_bar = sum_i alpha_i x_i` (Theorem 1, Appendix A, Eq. 24);
* **upper bound** `(1 - abar_N) d_1 + max_{j in I_N} ||x_j|| (abar_N (n-N) - (1 - abar_N))` (Theorem 1);
* **N_s/N** with `r = min_{i not in I_N} ||s - alpha_i x_i||` on vectors rescaled to the sphere of radius `M = 1`,
  `delta` = minimum pairwise distance among the row's vectors, and the two bounds of Theorem 2.

Here `n = t + 1` is the number of keys the row attends to. Rows with `n <= N` have no non-selected tokens and are skipped.
The value for a context length `L` is the mean over rows `t < L`; with causal attention and absolute positions these rows
are identical to those of a separate run of length `L`, so one 1024-token pass covers all `L`.
All quantities for N = 1..256 are obtained from a single cumulative-sum pass per row.

By default `x_i` are the per-head value vectors; `--vectors input` uses the hidden state entering the block instead.

The critical top-N for a given `L` is the smallest `N` whose per-head mean true distances and mean expectations
(144 values each) are not distinguished by a two-sample KS test at level `alpha` (0.01 by default).

The gradient experiment draws one random unit direction `v` per query row over the valid keys, averages
`||softmax((l + eps v)/T) - softmax(l/T)|| / eps` over rows and reports the maximum over heads, where `l` are the
scaled dot-product scores of GPT-2.

## Tests

```
PYTHONPATH=. python tests/test_metrics.py [path/to/gpt2-like/model]
```

compares the vectorised curves with a direct implementation of every formula for all `N`,
and the extracted attention weights with `output_attentions=True`.

## Citation

```
@article{mudarisov2025limitations,
  title   = {Limitations of Normalization in Attention Mechanism},
  author  = {Mudarisov, Timur and Burtsev, Mikhail and Petrova, Tatiana and State, Radu},
  journal = {arXiv preprint arXiv:2508.17821},
  year    = {2025}
}
```
