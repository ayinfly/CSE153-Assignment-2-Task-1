"""
Event-based Markov models for symbolic piano generation.

Instead of predicting one note at a time, this file trains Markov models over
onset events, where each event can contain multiple notes starting together.
"""

import random
from collections import Counter, defaultdict


def event_to_token(event):
    return (
        int(event["gap"]),
        tuple(event["pitches"]),
        tuple(event["durations"])
    )


def token_to_event(token):
    gap, pitches, durations = token

    return {
        "gap": gap,
        "pitches": pitches,
        "durations": durations
    }


def sample_from_probs(probs, temperature=1.0):
    items = list(probs.keys())
    weights = list(probs.values())

    if temperature != 1.0:
        weights = [w ** (1 / temperature) for w in weights]

    total = sum(weights)
    weights = [w / total for w in weights]

    return random.choices(items, weights=weights, k=1)[0]


def train_event_unigram(songs):
    counts = Counter()

    for song in songs:
        for event in song["events"]:
            counts[event_to_token(event)] += 1

    total = sum(counts.values())

    return {
        token: count / total
        for token, count in counts.items()
    }


def train_event_bigram(songs):
    counts = defaultdict(Counter)

    for song in songs:
        tokens = [event_to_token(event) for event in song["events"]]

        for prev_token, next_token in zip(tokens[:-1], tokens[1:]):
            counts[prev_token][next_token] += 1

    model = {}

    for prev_token, next_counts in counts.items():
        total = sum(next_counts.values())
        model[prev_token] = {
            token: count / total
            for token, count in next_counts.items()
        }

    return model


def train_event_second_order(songs):
    counts = defaultdict(Counter)

    for song in songs:
        tokens = [event_to_token(event) for event in song["events"]]

        for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
            counts[(a, b)][c] += 1

    model = {}

    for prev_pair, next_counts in counts.items():
        total = sum(next_counts.values())
        model[prev_pair] = {
            token: count / total
            for token, count in next_counts.items()
        }

    return model


def generate_event_unigram(event_unigram, n=200, temperature=1.0):
    tokens = [
        sample_from_probs(event_unigram, temperature=temperature)
        for _ in range(n)
    ]

    return [token_to_event(token) for token in tokens]


def generate_event_bigram(
    event_bigram,
    event_unigram,
    n=200,
    temperature=1.0,
    start=None
):
    if start is None:
        current = sample_from_probs(event_unigram, temperature=temperature)
    else:
        current = start

    output = [current]

    for _ in range(n - 1):
        if current in event_bigram:
            current = sample_from_probs(event_bigram[current], temperature=temperature)
        else:
            current = sample_from_probs(event_unigram, temperature=temperature)

        output.append(current)

    return [token_to_event(token) for token in output]


def generate_event_second_order(
    event_second_order,
    event_bigram,
    event_unigram,
    n=200,
    temperature=1.0
):
    first = sample_from_probs(event_unigram, temperature=temperature)

    if first in event_bigram:
        second = sample_from_probs(event_bigram[first], temperature=temperature)
    else:
        second = sample_from_probs(event_unigram, temperature=temperature)

    output = [first, second]

    for _ in range(n - 2):
        key = (output[-2], output[-1])

        if key in event_second_order:
            next_token = sample_from_probs(
                event_second_order[key],
                temperature=temperature
            )
        elif output[-1] in event_bigram:
            next_token = sample_from_probs(
                event_bigram[output[-1]],
                temperature=temperature
            )
        else:
            next_token = sample_from_probs(event_unigram, temperature=temperature)

        output.append(next_token)

    return [token_to_event(token) for token in output]