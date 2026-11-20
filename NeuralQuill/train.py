"""
train.py  —  Train the Shakespeare Transformer
Usage:
  python3 train.py               # default settings
  python3 train.py --iters 10000 # more training
  python3 train.py --help        # all options
"""

import argparse, os, sys, time, json, math, random
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model import CharTokenizer, ShakespeareTransformer


def get_batch(data, block_size, batch_size):
    """Sample a random batch of sequences."""
    ix = [random.randint(0, len(data) - block_size - 1)
          for _ in range(batch_size)]
    x  = [data[i : i + block_size]     for i in ix]
    y  = [data[i+1 : i + block_size+1] for i in ix]
    return x, y


def estimate_loss(model, train_data, val_data, block_size, eval_iters=50):
    """Average loss over several random batches."""
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size=1)
            xarr  = np.array(x[0], dtype=np.int32)
            yarr  = np.array(y[0], dtype=np.int32)
            _, loss = model.forward(xarr, targets=yarr)
            losses.append(float(loss))
        out[split] = np.mean(losses)
    return out


def train(args):
    DATA_PATH = os.path.join(BASE, 'input.txt')
    TOK_PATH  = os.path.join(BASE, 'tokenizer.pkl')
    MDL_PATH  = os.path.join(BASE, 'model.pkl')
    LOG_PATH  = os.path.join(BASE, 'loss_log.json')

    # ── Load data ────────────────────────────────────────────
    print("\n" + "═"*54)
    print("  Shakespeare Transformer  —  Training from Scratch")
    print("  NumPy only · No PyTorch · No GPU needed")
    print("═"*54)

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: {DATA_PATH} not found.")
        print("Make sure input.txt is in the same folder.")
        sys.exit(1)

    with open(DATA_PATH, encoding='utf-8') as f:
        text = f.read()
    print(f"\n[Data] {len(text):,} characters loaded")

    # ── Tokenizer ────────────────────────────────────────────
    print("[Step 1] Building character tokenizer...")
    tok = CharTokenizer()
    tok.build(text)
    tok.save(TOK_PATH)
    print(f"         Vocab: {tok.vocab_size} unique characters")

    # ── Encode ───────────────────────────────────────────────
    data = np.array(tok.encode(text), dtype=np.int32)
    n    = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"         Train tokens: {len(train_data):,}")
    print(f"         Val   tokens: {len(val_data):,}")

    # ── Model ────────────────────────────────────────────────
    print(f"\n[Step 2] Building model...")
    print(f"         n_embd={args.n_embd}  n_head={args.n_head}"
          f"  n_layer={args.n_layer}  block_size={args.block_size}")
    model = ShakespeareTransformer(
        vocab_size  = tok.vocab_size,
        n_embd      = args.n_embd,
        n_head      = args.n_head,
        n_layer     = args.n_layer,
        block_size  = args.block_size,
    )

    # ── Training loop ────────────────────────────────────────
    print(f"\n[Step 3] Training for {args.iters} iterations...")
    print(f"         lr={args.lr}  block_size={args.block_size}\n")

    best_val  = float('inf')
    loss_log  = {'train': [], 'val': [], 'iter': []}
    t_start   = time.time()

    for it in range(1, args.iters + 1):
        # sample one training sequence
        x, y   = get_batch(train_data, args.block_size, batch_size=1)
        xarr   = np.array(x[0], dtype=np.int32)
        yarr   = np.array(y[0], dtype=np.int32)

        # forward + backward
        model.P.zero_grad()
        _, loss = model.forward(xarr, targets=yarr)
        model.backward()

        # lr schedule: warmup then cosine decay
        warmup = min(500, args.iters // 10)
        if it <= warmup:
            cur_lr = args.lr * it / warmup
        else:
            prog   = (it - warmup) / (args.iters - warmup)
            cur_lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))

        model.P.step(lr=cur_lr)

        # eval + logging
        if it % args.eval_every == 0 or it == 1 or it == args.iters:
            losses = estimate_loss(model, train_data, val_data,
                                   args.block_size, eval_iters=30)
            loss_log['iter'].append(it)
            loss_log['train'].append(round(losses['train'], 4))
            loss_log['val'].append(round(losses['val'],   4))

            elapsed = time.time() - t_start
            filled  = int(24 * it / args.iters)
            bar     = '█' * filled + '░' * (24 - filled)
            print(f"  [{bar}] {it:5d}/{args.iters}"
                  f"  train={losses['train']:.4f}"
                  f"  val={losses['val']:.4f}"
                  f"  lr={cur_lr:.5f}"
                  f"  {elapsed:.0f}s")

            # save best
            if losses['val'] < best_val:
                best_val = losses['val']
                model.save(MDL_PATH)

        # sample preview every N iters
        if it % args.sample_every == 0:
            _show_sample(model, tok, args.block_size)

    # ── Done ─────────────────────────────────────────────────
    total_time = time.time() - t_start
    with open(LOG_PATH, 'w') as f: json.dump(loss_log, f)

    print(f"\n  ✓ Training complete!")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Total time    : {total_time:.1f}s")
    print(f"  Tokenizer  →  {TOK_PATH}")
    print(f"  Model      →  {MDL_PATH}")
    print(f"  Loss log   →  {LOG_PATH}")
    print(f"\n  Run app:  python3 app.py")
    print(f"  Generate: python3 generate.py --prompt \"HAMLET:\"")


def _show_sample(model, tok, block_size, length=120):
    """Print a short generated sample during training."""
    prompts = ["HAMLET:\n", "ROMEO:\n", "To be, or"]
    p       = random.choice(prompts)
    ids     = tok.encode(p)[-block_size:]
    gen     = model.generate(ids, max_new=length,
                              temperature=0.8, top_k=40)
    text    = tok.decode(gen)
    print(f"\n  ── Sample ─────────────────────────────────────")
    print(f"  {text[:200].strip()}")
    print(f"  ───────────────────────────────────────────────\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train Shakespeare Transformer')
    p.add_argument('--iters',        type=int,   default=5000)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--n_embd',       type=int,   default=128)
    p.add_argument('--n_head',       type=int,   default=4)
    p.add_argument('--n_layer',      type=int,   default=4)
    p.add_argument('--block_size',   type=int,   default=128)
    p.add_argument('--eval_every',   type=int,   default=500)
    p.add_argument('--sample_every', type=int,   default=1000)
    args = p.parse_args()
    train(args)
