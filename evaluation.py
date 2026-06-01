from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from markov import event_to_token

@torch.no_grad()
def neural_event_perplexity(model, tokenizer, songs, device=None, max_ctx=None):
    device = device or next(model.parameters()).device
    model.eval()
    is_transformer = hasattr(model, "cfg") and hasattr(model.cfg, "block_size")
    block = model.cfg.block_size if is_transformer else None

    total_nll, total_events, total_tokens = 0.0, 0, 0

    for song in songs:
        # tokens contributed by each event, in order
        per_event = [tokenizer.encode_event(e) for e in song["events"]]
        ids = [tokenizer.bos_id] + [t for ev in per_event for t in ev] + [tokenizer.eos_id]
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)

        # log p(id[t] | id[:t]) for t = 1 .. len-1
        logprobs = _sequence_logprobs(model, ids_t, is_transformer, block)
        # logprobs[j] is the log-prob of predicting ids[j+1]; i.e. it aligns to
        # target token ids[1], ids[2], ...
        cursor = 0  # index into target stream (0 -> ids[1])
        for ev_tokens in per_event:
            L = len(ev_tokens)
            nll = -logprobs[cursor:cursor + L].sum().item()
            total_nll += nll
            total_events += 1
            total_tokens += L
            cursor += L

    mean_nll_event = total_nll / max(total_events, 1)
    mean_nll_token = total_nll / max(total_tokens, 1)
    return {
        "nll_per_event": mean_nll_event,
        "perplexity_per_event": math.exp(mean_nll_event),
        "nll_per_token": mean_nll_token,
        "perplexity_per_token": math.exp(mean_nll_token),
        "num_events": total_events,
        "num_tokens": total_tokens,
    }


def _sequence_logprobs(model, ids_t, is_transformer, block):
    if is_transformer:
        T = ids_t.size(0)
        if T <= block:
            logits, _ = model(ids_t.unsqueeze(0))
            logp = F.log_softmax(logits[0], dim=-1)
            tgt = ids_t[1:]
            return logp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)

        # long sequence: slide a window of size `block`, scoring the tail.
        stride = block // 2
        out = torch.empty(T - 1, device=ids_t.device)
        # first window: score targets at positions 1..block-1
        logits, _ = model(ids_t[:block].unsqueeze(0))
        logp = F.log_softmax(logits[0], dim=-1)
        tgt = ids_t[1:block]
        out[0:block - 1] = logp[:block - 1].gather(1, tgt.unsqueeze(1)).squeeze(1)
        filled = block - 1  # number of targets scored so far (targets index from 1)

        # subsequent windows
        win_start = stride
        while filled < T - 1:
            win_end = min(win_start + block, T)
            chunk = ids_t[win_start:win_end]
            logits, _ = model(chunk.unsqueeze(0))
            logp = F.log_softmax(logits[0], dim=-1)
            # targets we still need start at absolute position `filled+1`;
            # within this window that is local index (filled + 1 - win_start).
            first_local_tgt = filled + 1 - win_start
            local_tgt = ids_t[win_start + first_local_tgt: win_end]
            picked = logp[first_local_tgt - 1: win_end - win_start - 1].gather(
                1, local_tgt.unsqueeze(1)).squeeze(1)
            n_new = picked.numel()
            out[filled: filled + n_new] = picked
            filled += n_new
            win_start += stride
        return out
    else:
        logits, _ = model(ids_t.unsqueeze(0))
        logp = F.log_softmax(logits[0], dim=-1)
        tgt = ids_t[1:]
        return logp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)


def markov_event_perplexity(
    songs, unigram, bigram=None, second_order=None, floor=1e-6, tokenizer=None,
):
    total_nll, total_events, total_tokens = 0.0, 0, 0

    for song in songs:
        tokens = [event_to_token(e) for e in song["events"]]
        for i, tok in enumerate(tokens):
            p = None
            if second_order is not None and i >= 2:
                key = (tokens[i - 2], tokens[i - 1])
                if key in second_order and tok in second_order[key]:
                    p = second_order[key][tok]
            if p is None and bigram is not None and i >= 1:
                prev = tokens[i - 1]
                if prev in bigram and tok in bigram[prev]:
                    p = bigram[prev][tok]
            if p is None:
                p = unigram.get(tok, floor)
            p = max(p, floor)
            total_nll += -math.log(p)
            total_events += 1
            if tokenizer is not None:
                total_tokens += len(tokenizer.encode_event(song["events"][i]))

    mean_nll_event = total_nll / max(total_events, 1)
    out = {"nll_per_event": mean_nll_event,
           "perplexity_per_event": math.exp(mean_nll_event),
           "num_events": total_events}
    if tokenizer is not None:
        mean_nll_token = total_nll / max(total_tokens, 1)
        out["nll_per_token"] = mean_nll_token
        out["perplexity_per_token"] = math.exp(mean_nll_token)
        out["num_tokens"] = total_tokens
    return out


def _all_pitches(events):
    return [p for e in events for p in e["pitches"]]


def pitch_class_histogram(events):
    h = np.zeros(12)
    for p in _all_pitches(events):
        h[int(p) % 12] += 1
    s = h.sum()
    return h / s if s > 0 else h


def kl_divergence(p, q, eps=1e-8):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def avg_polyphony(events):
    sizes = [len(e["pitches"]) for e in events if len(e["pitches"]) > 0]
    return float(np.mean(sizes)) if sizes else 0.0


# Grid resolution of the preprocessing: the ticks-per-beat used when the events were quantised. Only the *relative* density across models matters, but we name the constant so it isn't a mystery literal. Set this to match your preprocessing.
GRID = 120

def note_density(events):
    total_notes = sum(len(e["pitches"]) for e in events)
    total_time = sum(max(0, int(e["gap"])) for e in events)
    return float(total_notes / total_time * GRID) if total_time > 0 else 0.0


def pitch_range(events):
    ps = _all_pitches(events)
    return (min(ps), max(ps)) if ps else (0, 0)


def repetition_rate(events):
    toks = [event_to_token(e) for e in events]
    if len(toks) < 2:
        return 0.0
    reps = sum(1 for a, b in zip(toks[:-1], toks[1:]) if a == b)
    return reps / (len(toks) - 1)


def distinct_event_ratio(events):
    toks = [event_to_token(e) for e in events]
    return len(set(toks)) / len(toks) if toks else 0.0


# Krumhansl-Kessler key profiles for scale-consistency scoring.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_consistency(events):
    hist = pitch_class_histogram(events)
    if hist.sum() == 0:
        return 0.0
    best = -1.0
    for prof in (_MAJOR, _MINOR):
        for shift in range(12):
            rotated = np.roll(prof, shift)
            c = np.corrcoef(hist, rotated)[0, 1]
            if c > best:
                best = c
    return float(best)


def musical_report(events, reference_events):
    return {
        "num_events": len(events),
        "num_notes": sum(len(e["pitches"]) for e in events),
        "avg_polyphony": round(avg_polyphony(events), 3),
        "note_density": round(note_density(events), 4),
        "pitch_range": pitch_range(events),
        "repetition_rate": round(repetition_rate(events), 3),
        "distinct_event_ratio": round(distinct_event_ratio(events), 3),
        "key_consistency": round(key_consistency(events), 3),
        "pc_kl_vs_reference": round(
            kl_divergence(pitch_class_histogram(events),
                          pitch_class_histogram(reference_events)), 4),
    }


def reference_stats(songs):
    events = [e for s in songs for e in s["events"]]
    return {
        "avg_polyphony": round(avg_polyphony(events), 3),
        "note_density": round(note_density(events), 4),
        "repetition_rate": round(repetition_rate(events), 3),
        "distinct_event_ratio": round(distinct_event_ratio(events), 3),
        "key_consistency": round(key_consistency(events), 3),
        "pc_kl_vs_reference": 0.0,
        "num_events": len(events),
        "num_notes": sum(len(e["pitches"]) for e in events),
        "pitch_range": pitch_range(events),
    }


def aggregate_reports(samples, reference_events):
    reports = [musical_report(s, reference_events) for s in samples]
    num_keys = ["avg_polyphony", "note_density", "repetition_rate",
                "distinct_event_ratio", "key_consistency", "pc_kl_vs_reference"]
    out = {}
    for k in num_keys:
        v = np.array([r[k] for r in reports], dtype=float)
        out[k] = f"{v.mean():.3f} +/- {v.std():.3f}"
    out["num_events"] = int(np.mean([r["num_events"] for r in reports]))
    out["num_notes"]  = int(np.mean([r["num_notes"] for r in reports]))
    mins = [r["pitch_range"][0] for r in reports]
    maxs = [r["pitch_range"][1] for r in reports]
    out["pitch_range"] = (min(mins), max(maxs))
    return out