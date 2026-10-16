import os
import urllib.request

WAR_AND_PEACE = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
START = "Well, Prince, so Genoa"


def read_text(path=None, cache="data/war_and_peace.txt"):
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        urllib.request.urlretrieve(WAR_AND_PEACE, cache)
    with open(cache, encoding="utf-8") as f:
        text = f.read()
    end = text.find("*** END OF")
    if end > 0:
        text = text[:end]
    start = text.find(START)
    if start >= 0:
        return text[text.rfind("\n", 0, start) + 1 :]
    start = text.find("*** START OF")
    return text[text.find("\n", start) + 1 :] if start >= 0 else text


def token_chunks(tok, text, length, n_chunks=1, offset=0):
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = []
    for c in range(n_chunks):
        a = offset + c * length
        if a + length > len(ids):
            break
        chunks.append(ids[a : a + length])
    return chunks
