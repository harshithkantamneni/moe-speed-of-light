"""The case-study bound must charge the LM head to dense (non-expert) work, as the decode model does (erratum, 28 Sep 2026)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import sol_a10, sol_windows  # noqa: E402

G = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "gguf_bytes.json")))


def test_dense_bytes_include_head():
    for tag, key in sol_a10.GGUF.items():
        want = int(G[key]["dense_bytes"] + G[key]["head_bytes"])
        assert sol_a10.MODELS[tag][7] == want, tag
        assert sol_windows.MODELS[tag][6] == want, tag


if __name__ == "__main__":
    test_dense_bytes_include_head()
    print("ok")
