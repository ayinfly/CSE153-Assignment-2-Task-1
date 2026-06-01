"""
Neural sequence models for symbolic, unconditioned piano generation.

This module is the neural counterpart to markov.py. The Markov models in that
file treat each whole onset event -- the tuple (gap, pitches, durations) -- as a
single atomic token. That works for counting n-grams but is hopeless for a
neural network: almost every distinct chord+timing combination appears only
once, so the vocabulary is enormous and most tokens are unlearnable.

Here we instead DECOMPOSE every event into a short sequence of sub-tokens:

    GAP_g  PITCH_p1 DUR_d1  PITCH_p2 DUR_d2  ...  EOE

and train an autoregressive model over the flattened sub-token stream. Because
gaps and durations are already quantised to a coarse grid in preprocessing.py
(multiples of 120, capped at 1920) and pitches live in [21, 108], the resulting
vocabulary is only a few hundred symbols -- each seen many times -- which is
exactly what a neural model needs.

Two models are provided:
  * EventLSTM         -- a recurrent baseline (warm-up; quick to train/debug).
  * EventTransformer  -- a small from-scratch decoder-only Transformer (the main
                         model), with a correct causal mask.

Both are trained from scratch (random init), satisfying the assignment
requirement to "fit your own model weights" rather than load a checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GRID = 120
MAX_GAP = 1920
MAX_DURATION = 1920
MIN_PITCH = 21
MAX_PITCH = 108

SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "EOE"]  # EOE = end-of-event


class EventTokenizer:

    def __init__(self):
        tokens = list(SPECIAL_TOKENS)
        tokens += [f"GAP_{g}" for g in range(0, MAX_GAP + 1, GRID)]
        tokens += [f"DUR_{d}" for d in range(GRID, MAX_DURATION + 1, GRID)]
        tokens += [f"PITCH_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]

        self.itos = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}

        self.pad_id = self.stoi["PAD"]
        self.bos_id = self.stoi["BOS"]
        self.eos_id = self.stoi["EOS"]
        self.eoe_id = self.stoi["EOE"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _snap_gap(self, g):
        g = int(round(g / GRID) * GRID)
        return min(max(g, 0), MAX_GAP)

    def _snap_dur(self, d):
        d = int(round(d / GRID) * GRID)
        return min(max(d, GRID), MAX_DURATION)

    def encode_event(self, event) -> list[int]:
        """One event dict -> [GAP, (PITCH,DUR)*, EOE] ids."""
        ids = [self.stoi[f"GAP_{self._snap_gap(event['gap'])}"]]
        for pitch, dur in zip(event["pitches"], event["durations"]):
            pitch = int(pitch)
            if not (MIN_PITCH <= pitch <= MAX_PITCH):
                continue  # drop unplayable pitches (midi_output would too)
            ids.append(self.stoi[f"PITCH_{pitch}"])
            ids.append(self.stoi[f"DUR_{self._snap_dur(dur)}"])
        ids.append(self.eoe_id)
        return ids

    def encode_song(self, song, add_bos_eos=True) -> list[int]:
        ids = [self.bos_id] if add_bos_eos else []
        for event in song["events"]:
            ids.extend(self.encode_event(event))
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def encode_songs(self, songs) -> list[int]:
        stream = []
        for song in songs:
            stream.extend(self.encode_song(song, add_bos_eos=True))
        return stream

    def decode_to_events(self, ids) -> list[dict]:
        events = []
        cur_gap = None
        cur_pitches = []
        cur_durs = []
        pending_pitch = None

        def flush():
            nonlocal cur_gap, cur_pitches, cur_durs, pending_pitch
            if cur_gap is not None and cur_pitches:
                events.append({
                    "gap": cur_gap,
                    "pitches": tuple(cur_pitches),
                    "durations": tuple(cur_durs),
                })
            cur_pitches, cur_durs, pending_pitch = [], [], None

        for i in ids:
            tok = self.itos[int(i)]
            if tok in ("PAD", "BOS"):
                continue
            if tok == "EOS":
                break
            if tok.startswith("GAP_"):
                flush()                       # close previous event
                cur_gap = int(tok[4:])
                pending_pitch = None
            elif tok == "EOE":
                flush()
                cur_gap = None
            elif tok.startswith("PITCH_"):
                pending_pitch = int(tok[6:])
            elif tok.startswith("DUR_"):
                if pending_pitch is not None and cur_gap is not None:
                    cur_pitches.append(pending_pitch)
                    cur_durs.append(int(tok[4:]))
                    pending_pitch = None
        flush()
        return events


def make_windows(stream, block_size, stride=None):
    stream = torch.as_tensor(stream, dtype=torch.long)
    stride = stride or block_size
    if len(stream) < block_size + 1:
        raise ValueError(
            f"Stream too short ({len(stream)}) for block_size {block_size}."
        )
    # max start s.t. a length-block_size window AND its +1 target both fit
    starts = list(range(0, len(stream) - block_size, stride))
    if not starts:
        starts = [0]
    X = torch.stack([stream[i:i + block_size] for i in starts])
    Y = torch.stack([stream[i + 1:i + block_size + 1] for i in starts])
    return X, Y


def iterate_batches(X, Y, batch_size, shuffle=True, generator=None):
    n = X.size(0)
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        sel = idx[s:s + batch_size]
        yield X[sel], Y[sel]


def transpose_events(events, semitones):
    if semitones == 0:
        return list(events)
    out = []
    carry_gap = 0
    for e in events:
        ps, ds = [], []
        for p, d in zip(e["pitches"], e["durations"]):
            q = int(p) + semitones
            if MIN_PITCH <= q <= MAX_PITCH:
                ps.append(q)
                ds.append(d)
        if ps:
            out.append({"gap": int(e["gap"]) + carry_gap,
                        "pitches": tuple(ps), "durations": tuple(ds)})
            carry_gap = 0
        else:
            carry_gap += int(e["gap"])
    return out


def encode_songs_augmented(tokenizer, songs, shifts=(0,)):
    stream = []
    for s in shifts:
        for song in songs:
            ev = transpose_events(song["events"], s)
            if ev:
                stream.extend(tokenizer.encode_song({"events": ev}, add_bos_eos=True))
    return stream


@dataclass
class LSTMConfig:
    vocab_size: int
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2


class EventLSTM(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.lstm = nn.LSTM(
            cfg.embed_dim, cfg.hidden_dim, num_layers=cfg.num_layers,
            batch_first=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x, hidden=None):
        emb = self.drop(self.embed(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(self.drop(out))
        return logits, hidden


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # lower-triangular mask, registered as a buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask)
        # expose the attention weights from the last forward for visualisation
        self.last_attn = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)   # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)     # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()
        att = self.attn_drop(att)

        y = att @ v                                         # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm residual
        x = x + self.mlp(self.ln2(x))
        return x


class EventTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying improves quality and saves params
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        tok = self.tok_emb(x)
        pos = self.pos_emb[:, :T, :]
        h = self.drop(tok + pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h), None   # (logits, _) to match LSTM's signature

    def attention_maps(self):
        """List of last-forward attention tensors, one per layer."""
        return [blk.attn.last_attn for blk in self.blocks]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _cosine_warmup_factor(step, total_steps, warmup_steps):
    """LR multiplier in [0, 1]: linear warmup then cosine decay to ~0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_model(
    model, train_stream, val_stream, tokenizer, block_size,
    epochs=10, batch_size=64, lr=3e-4, weight_decay=0.01,
    device=None, log_every=1, seed=0,
    train_stride=None, lr_schedule=None, warmup_frac=0.05,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = model.to(device)

    Xtr, Ytr = make_windows(train_stream, block_size, stride=train_stride)
    Xva, Yva = make_windows(val_stream, block_size) if len(val_stream) > block_size + 1 else (None, None)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_schedule == "cosine":
        nb_per_epoch = (Xtr.size(0) + batch_size - 1) // batch_size
        total_steps = max(1, epochs * nb_per_epoch)
        warmup_steps = int(warmup_frac * total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: _cosine_warmup_factor(s, total_steps, warmup_steps)
        )

    gen = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_loss": [], "val_ppl_token": []}

    # Track the best-validation checkpoint so we can restore it at the end
    # instead of keeping the final (possibly over-fit) weights.
    best_val, best_state, best_epoch = float("inf"), None, None

    for epoch in range(1, epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for xb, yb in iterate_batches(Xtr, Ytr, batch_size, shuffle=True, generator=gen):
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item(); nb += 1
        tr_loss = running / max(nb, 1)
        history["train_loss"].append(tr_loss)

        va_loss = float("nan")
        if Xva is not None:
            va_loss = evaluate_loss(model, Xva, Yva, tokenizer, batch_size, device)
        history["val_loss"].append(va_loss)
        history["val_ppl_token"].append(math.exp(va_loss) if va_loss == va_loss else float("nan"))

        # keep a CPU copy of the weights at the best validation loss so far
        if va_loss == va_loss and va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0:
            msg = f"epoch {epoch:>3} | train loss {tr_loss:.4f}"
            if va_loss == va_loss:
                msg += f" | val loss {va_loss:.4f} | val ppl/token {math.exp(va_loss):.2f}"
            print(msg)

    # restore the best-validation weights (early-stopping by checkpoint)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch} "
              f"(val loss {best_val:.4f}, val ppl/token {math.exp(best_val):.2f})")
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val

    return history


@torch.no_grad()
def evaluate_loss(model, X, Y, tokenizer, batch_size=64, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    tot, ntok = 0.0, 0
    for xb, yb in iterate_batches(X, Y, batch_size, shuffle=False):
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), yb.reshape(-1),
            ignore_index=tokenizer.pad_id, reduction="sum",
        )
        tot += loss.item()
        ntok += (yb != tokenizer.pad_id).sum().item()
    return tot / max(ntok, 1)


@torch.no_grad()
def generate(
    model, tokenizer, max_tokens=600, temperature=1.0, top_k=None, top_p=None,
    device=None, seed=None, prompt_ids=None, stop_at_eos=True,
):
    """Returns a list of event dicts ready for save_event_midi."""
    device = device or next(model.parameters()).device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    is_transformer = isinstance(model, EventTransformer)
    block_size = model.cfg.block_size if is_transformer else None

    ids = list(prompt_ids) if prompt_ids else [tokenizer.bos_id]
    hidden = None

    for _ in range(max_tokens):
        if is_transformer:
            ctx = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
            logits, _ = model(ctx)
            logits = logits[0, -1]
        else:
            # feed only the newest token, carrying LSTM hidden state forward
            step = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
            logits, hidden = model(step, hidden)
            logits = logits[0, -1]

        logits = logits / max(temperature, 1e-6)
        logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        ids.append(nxt)
        if stop_at_eos and nxt == tokenizer.eos_id:
            break

    return tokenizer.decode_to_events(ids)


def _filter_logits(logits, top_k=None, top_p=None):
    logits = logits.clone()
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_idx[remove]] = float("-inf")
    return logits