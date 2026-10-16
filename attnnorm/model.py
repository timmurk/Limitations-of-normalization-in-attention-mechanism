import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gpt2(name="gpt2", device="cpu"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager")
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def attention_tensors(model, ids):
    blocks = model.transformer.h
    n_head = model.config.n_head
    store, handles = {}, []

    def qkv_hook(i):
        def hook(mod, args, out):
            store[("qkv", i)] = out.detach()
        return hook

    def input_hook(i):
        def hook(mod, args):
            store[("input", i)] = args[0].detach()
        return hook

    for i, block in enumerate(blocks):
        handles.append(block.attn.c_attn.register_forward_hook(qkv_hook(i)))
        handles.append(block.register_forward_pre_hook(input_hook(i)))
    model(input_ids=ids[None], use_cache=False)
    for h in handles:
        h.remove()

    layers = []
    for i in range(len(blocks)):
        qkv = store[("qkv", i)][0]
        L = qkv.shape[0]
        d = qkv.shape[1] // 3
        dh = d // n_head
        q, k, v = (x.view(L, n_head, dh).transpose(0, 1) for x in qkv.split(d, dim=1))
        scores = q @ k.transpose(1, 2) / dh ** 0.5
        mask = torch.ones(L, L, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        layers.append(dict(
            scores=scores,
            attn=torch.softmax(scores, -1),
            value=v,
            input=store[("input", i)][0],
        ))
    return layers
