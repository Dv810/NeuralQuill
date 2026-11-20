"""
generate.py  —  Generate Shakespeare-style text
Usage:
  python3 generate.py
  python3 generate.py --prompt "HAMLET:"
  python3 generate.py --prompt "To be, or" --length 500 --temp 0.9
  python3 generate.py --temp 0.6 --top_k 20   # more conservative
"""

import argparse, os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model import CharTokenizer, ShakespeareTransformer

TOK_PATH = os.path.join(BASE, 'tokenizer.pkl')
MDL_PATH = os.path.join(BASE, 'model.pkl')

PROMPTS = [
    "HAMLET:\n",
    "ROMEO:\n",
    "MACBETH:\n",
    "JULIET:\n",
    "KING LEAR:\n",
    "PROSPERO:\n",
    "To be, or not to be,",
    "Friends, Romans,",
    "All the world's a stage,",
    "What light through yonder",
]


def generate(prompt, length=400, temperature=0.8, top_k=40):
    if not os.path.exists(MDL_PATH):
        print("ERROR: model.pkl not found. Run train.py first.")
        sys.exit(1)

    tok = CharTokenizer.load(TOK_PATH)
    model = ShakespeareTransformer(
        vocab_size=tok.vocab_size,
        n_embd=128, n_head=4, n_layer=4, block_size=128
    )
    model.load(MDL_PATH)

    ids  = tok.encode(prompt)[-128:]
    gen  = model.generate(ids, max_new=length,
                          temperature=temperature, top_k=top_k)
    return tok.decode(gen)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', type=str,   default=None)
    p.add_argument('--length', type=int,   default=400)
    p.add_argument('--temp',   type=float, default=0.8)
    p.add_argument('--top_k',  type=int,   default=40)
    args = p.parse_args()

    import random
    prompt = args.prompt or random.choice(PROMPTS)

    print("\n" + "═"*56)
    print(f"  Prompt: {repr(prompt)}")
    print(f"  Temp:   {args.temp}   Top-k: {args.top_k}")
    print("═"*56 + "\n")

    text = generate(prompt, args.length, args.temp, args.top_k)
    print(text)
    print("\n" + "═"*56)
