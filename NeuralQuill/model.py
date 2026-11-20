"""
model.py  —  Character-level Transformer, built from scratch
             NumPy only. No PyTorch. No APIs. No dependencies.

Architecture:
  - Character-level tokenizer
  - Token + Positional embeddings
  - N x Transformer blocks
      Multi-Head Causal Self-Attention
      Feed-Forward Network
      Layer Norm + Residual connections
  - LM Head (linear projection to vocab)
  - Trained with cross-entropy + Adam
"""

import numpy as np
import math, pickle, random, os

np.random.seed(1337)
random.seed(1337)


# ══════════════════════════════════════════════════════════════
# TOKENIZER  (character level)
# ══════════════════════════════════════════════════════════════

class CharTokenizer:
    def __init__(self):
        self.chars    = []
        self.stoi     = {}   # char → int
        self.itos     = {}   # int  → char
        self.vocab_size = 0

    def build(self, text):
        self.chars      = sorted(list(set(text)))
        self.stoi       = {c: i for i, c in enumerate(self.chars)}
        self.itos       = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        print(f"[Tokenizer] Vocab: {self.vocab_size} chars")

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos.get(i, '') for i in ids)

    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f: return pickle.load(f)


# ══════════════════════════════════════════════════════════════
# MATH UTILS
# ══════════════════════════════════════════════════════════════

def softmax(x, axis=-1):
    x  = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=axis, keepdims=True) + 1e-9)

def relu(x):
    return np.maximum(0.0, x)

def layer_norm(x, g, b, eps=1e-5):
    mu  = x.mean(-1, keepdims=True)
    std = x.std (-1, keepdims=True) + eps
    return g * (x - mu) / std + b

def cross_entropy(logits, targets):
    """logits (T,V) targets (T,) → scalar loss, probs"""
    p = softmax(logits)
    p = np.clip(p, 1e-9, 1.0)
    T = len(targets)
    loss = -np.log(p[np.arange(T), targets]).mean()
    return loss, p


# ══════════════════════════════════════════════════════════════
# PARAMETER STORE  (weights + Adam state)
# ══════════════════════════════════════════════════════════════

class Params:
    def __init__(self):
        self.w={}; self.g={}; self.m={}; self.v={}; self.t=0

    def add(self, name, shape, scale=0.02):
        self.w[name] = (np.random.randn(*shape)*scale).astype(np.float32)
        self.g[name] = np.zeros(shape, np.float32)
        self.m[name] = np.zeros(shape, np.float32)
        self.v[name] = np.zeros(shape, np.float32)

    def zeros(self, name, shape):
        self.w[name] = np.zeros(shape, np.float32)
        self.g[name] = np.zeros(shape, np.float32)
        self.m[name] = np.zeros(shape, np.float32)
        self.v[name] = np.zeros(shape, np.float32)

    def ones(self, name, shape):
        self.w[name] = np.ones(shape, np.float32)
        self.g[name] = np.zeros(shape, np.float32)
        self.m[name] = np.zeros(shape, np.float32)
        self.v[name] = np.zeros(shape, np.float32)

    def zero_grad(self):
        for k in self.g: self.g[k][:] = 0

    def step(self, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.t += 1
        # gradient clipping
        norm = math.sqrt(sum(float(np.sum(gg**2)) for gg in self.g.values()))
        if norm > clip:
            scale = clip / (norm + 1e-8)
            for k in self.g: self.g[k] *= scale
        # adam
        for k in self.w:
            self.m[k] = b1*self.m[k] + (1-b1)*self.g[k]
            self.v[k] = b2*self.v[k] + (1-b2)*self.g[k]**2
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            self.w[k] -= lr * mh / (np.sqrt(vh) + eps)

    def count(self):
        return sum(a.size for a in self.w.values())

    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump({'w':self.w,'m':self.m,'v':self.v,'t':self.t}, f)

    def load(self, path):
        with open(path,'rb') as f: d = pickle.load(f)
        self.w=d['w']; self.m=d['m']; self.v=d['v']; self.t=d['t']
        self.g = {k: np.zeros_like(v) for k,v in self.w.items()}


# ══════════════════════════════════════════════════════════════
# TRANSFORMER
# ══════════════════════════════════════════════════════════════

class ShakespeareTransformer:
    """
    Decoder-only Transformer language model.
    Character level. Built entirely from scratch.
    """

    def __init__(self, vocab_size, n_embd=128, n_head=4,
                 n_layer=4, block_size=128):
        self.V  = vocab_size
        self.d  = n_embd
        self.h  = n_head
        self.hs = n_embd // n_head
        self.nl = n_layer
        self.T  = block_size
        self.P  = Params()
        self.C  = {}          # forward cache
        self._init_weights()

    def _init_weights(self):
        P  = self.P
        d  = self.d
        V  = self.V
        T  = self.T
        nl = self.nl
        h  = self.h
        hs = self.hs

        # embeddings
        P.add  ('tok', (V, d), scale=0.02)
        P.add  ('pos', (T, d), scale=0.02)

        for i in range(nl):
            p = f'l{i}'
            # attention projections  (fused QKV)
            P.add  (f'{p}_qkv', (d, 3*d), scale=0.02)
            P.add  (f'{p}_proj',(d, d),   scale=0.02)
            P.zeros(f'{p}_proj_b', (d,))
            # feed-forward
            P.add  (f'{p}_ff1', (d, 4*d), scale=0.02)
            P.zeros(f'{p}_ff1_b', (4*d,))
            P.add  (f'{p}_ff2', (4*d, d), scale=0.02)
            P.zeros(f'{p}_ff2_b', (d,))
            # layer norms
            P.ones (f'{p}_ln1g', (d,));  P.zeros(f'{p}_ln1b', (d,))
            P.ones (f'{p}_ln2g', (d,));  P.zeros(f'{p}_ln2b', (d,))

        P.ones ('lnfg', (d,)); P.zeros('lnfb', (d,))
        P.add  ('head', (d, V), scale=0.02)

        print(f"[Model] {self.P.count():,} parameters  "
              f"({self.P.count()/1e6:.2f}M)")

    # ── causal mask ──────────────────────────────────────────
    @staticmethod
    def _cmask(T):
        return np.tril(np.ones((T,T), bool))

    # ── layer-norm backward ──────────────────────────────────
    @staticmethod
    def _ln_bwd(dout, x, g, eps=1e-5):
        T, d = x.shape
        mu   = x.mean(-1, keepdims=True)
        std  = x.std (-1, keepdims=True) + eps
        xh   = (x - mu) / std
        dg   = (dout * xh).sum(0)
        db   = dout.sum(0)
        dxh  = dout * g
        dx   = (1/d)/std * (d*dxh
               - dxh.sum(-1, keepdims=True)
               - xh*(dxh*xh).sum(-1, keepdims=True))
        return dx, dg, db

    # ── forward ──────────────────────────────────────────────
    def forward(self, idx, targets=None):
        """
        idx:     (T,) int  — input token ids
        targets: (T,) int  — shifted targets (for training)
        returns: logits (T,V), loss or None
        """
        P    = self.P.w
        T    = len(idx)
        d    = self.d
        h    = self.h
        hs   = self.hs
        mask = self._cmask(T)
        C    = self.C = {}

        # embeddings
        tok  = P['tok'][idx]          # (T,d)
        pos  = P['pos'][:T]           # (T,d)
        x    = tok + pos
        C['x0']  = x.copy()
        C['idx'] = idx
        C['tok'] = tok

        # transformer blocks
        for i in range(self.nl):
            p = f'l{i}'
            C[f'{p}_xin'] = x.copy()

            # --- LayerNorm 1 ---
            ln1 = layer_norm(x, P[f'{p}_ln1g'], P[f'{p}_ln1b'])
            C[f'{p}_ln1'] = ln1

            # --- Multi-Head Attention (fused QKV) ---
            qkv = ln1 @ P[f'{p}_qkv']            # (T, 3d)
            Q, K, V = np.split(qkv, 3, axis=-1)  # each (T,d)
            C[f'{p}_Q']=Q; C[f'{p}_K']=K; C[f'{p}_V']=V

            # split into heads
            Q = Q.reshape(T, h, hs).transpose(1,0,2)  # (h,T,hs)
            K = K.reshape(T, h, hs).transpose(1,0,2)
            V = V.reshape(T, h, hs).transpose(1,0,2)

            scale = 1.0 / math.sqrt(hs)
            att   = (Q @ K.transpose(0,2,1)) * scale   # (h,T,T)
            att   = np.where(mask[None], att, -1e9)
            att   = softmax(att, axis=-1)               # (h,T,T)
            C[f'{p}_att'] = att

            out = att @ V                               # (h,T,hs)
            out = out.transpose(1,0,2).reshape(T, d)   # (T,d)
            C[f'{p}_out'] = out

            attn = out @ P[f'{p}_proj'] + P[f'{p}_proj_b']  # (T,d)
            C[f'{p}_attn'] = attn
            x = x + attn
            C[f'{p}_r1'] = x.copy()

            # --- LayerNorm 2 ---
            ln2 = layer_norm(x, P[f'{p}_ln2g'], P[f'{p}_ln2b'])
            C[f'{p}_ln2in'] = x.copy()
            C[f'{p}_ln2']   = ln2

            # --- Feed-Forward ---
            ff1 = relu(ln2 @ P[f'{p}_ff1'] + P[f'{p}_ff1_b'])  # (T,4d)
            ff2 = ff1  @ P[f'{p}_ff2'] + P[f'{p}_ff2_b']       # (T,d)
            C[f'{p}_ff1'] = ff1
            x = x + ff2
            C[f'{p}_r2'] = x.copy()

        # final LN + head
        C['xpre']   = x.copy()
        xf          = layer_norm(x, P['lnfg'], P['lnfb'])
        C['xf']     = xf
        logits      = xf @ P['head']                            # (T,V)
        C['logits'] = logits

        loss = None
        if targets is not None:
            loss, probs = cross_entropy(logits, targets)
            C['probs']   = probs
            C['targets'] = targets

        return logits, loss

    # ── backward ─────────────────────────────────────────────
    def backward(self):
        P  = self.P.w
        G  = self.P.g
        C  = self.C
        T  = len(C['idx'])
        d  = self.d
        h  = self.h
        hs = self.hs

        # dLogits
        probs   = C['probs'].copy()
        targets = C['targets']
        dl      = probs.copy()
        dl[np.arange(T), targets] -= 1
        dl /= T                                        # (T,V)

        # head
        G['head'] += C['xf'].T @ dl
        dx = dl @ P['head'].T                         # (T,d)

        # final LN
        dx, dg, db = self._ln_bwd(dx, C['xpre'], P['lnfg'])
        G['lnfg'] += dg; G['lnfb'] += db

        for i in reversed(range(self.nl)):
            p = f'l{i}'

            # residual 2
            dff2 = dx.copy()
            # ff2 backward
            G[f'{p}_ff2']   += C[f'{p}_ff1'].T @ dff2
            G[f'{p}_ff2_b'] += dff2.sum(0)
            dff1  = (dff2 @ P[f'{p}_ff2'].T) * (C[f'{p}_ff1'] > 0)
            G[f'{p}_ff1']   += C[f'{p}_ln2'].T @ dff1
            G[f'{p}_ff1_b'] += dff1.sum(0)
            dln2  = dff1 @ P[f'{p}_ff1'].T

            # LN2 backward
            dln2, dg, db = self._ln_bwd(dln2, C[f'{p}_ln2in'], P[f'{p}_ln2g'])
            G[f'{p}_ln2g'] += dg; G[f'{p}_ln2b'] += db
            dx = dx + dln2                            # residual

            # residual 1
            dattn = dx.copy()
            G[f'{p}_proj']   += C[f'{p}_out'].T @ dattn
            G[f'{p}_proj_b'] += dattn.sum(0)
            dout = dattn @ P[f'{p}_proj'].T           # (T,d)

            # multi-head attention backward
            dout_h = dout.reshape(T, h, hs).transpose(1,0,2)  # (h,T,hs)
            att    = C[f'{p}_att']                              # (h,T,T)
            Q  = C[f'{p}_Q'].reshape(T,h,hs).transpose(1,0,2)
            K  = C[f'{p}_K'].reshape(T,h,hs).transpose(1,0,2)
            V  = C[f'{p}_V'].reshape(T,h,hs).transpose(1,0,2)

            dV  = att.transpose(0,2,1) @ dout_h       # (h,T,hs)
            dat = dout_h @ V.transpose(0,2,1)          # (h,T,T)
            # softmax backward
            dat = att * (dat - (dat*att).sum(-1,keepdims=True))
            dat /= math.sqrt(hs)
            dQ  = dat @ K                              # (h,T,hs)
            dK  = dat.transpose(0,2,1) @ Q

            dQ = dQ.transpose(1,0,2).reshape(T,d)
            dK = dK.transpose(1,0,2).reshape(T,d)
            dV = dV.transpose(1,0,2).reshape(T,d)
            dqkv = np.concatenate([dQ, dK, dV], axis=-1)   # (T,3d)

            G[f'{p}_qkv'] += C[f'{p}_ln1'].T @ dqkv
            dln1 = dqkv @ P[f'{p}_qkv'].T

            # LN1 backward
            dln1, dg, db = self._ln_bwd(dln1, C[f'{p}_xin'], P[f'{p}_ln1g'])
            G[f'{p}_ln1g'] += dg; G[f'{p}_ln1b'] += db
            dx = dx + dln1                            # residual

        # embedding backward
        np.add.at(G['tok'], C['idx'], dx)
        G['pos'][:T] += dx

    # ── generate ─────────────────────────────────────────────
    @np.errstate(all='ignore')
    def generate(self, prompt_ids, max_new=500, temperature=0.8, top_k=40):
        """
        prompt_ids : list of int
        temperature: >1 = more creative, <1 = more conservative
        top_k      : only sample from top-k most likely chars
        """
        ids = list(prompt_ids)
        for _ in range(max_new):
            ctx     = np.array(ids[-self.T:], dtype=np.int32)
            logits, _ = self.forward(ctx)
            lg      = logits[-1].astype(np.float64)

            # temperature
            lg /= max(temperature, 1e-9)

            # top-k filtering
            if top_k > 0:
                thresh  = np.sort(lg)[-top_k]
                lg      = np.where(lg >= thresh, lg, -1e9)

            probs   = softmax(lg)
            probs   = np.nan_to_num(probs, nan=0.0)
            probs   = np.clip(probs, 0, None)
            s       = probs.sum()
            if s <= 0: probs = np.ones_like(probs)/len(probs)
            else:      probs /= s

            nxt = int(np.random.choice(len(probs), p=probs))
            ids.append(nxt)

        return ids

    def save(self, path): self.P.save(path)
    def load(self, path): self.P.load(path)
