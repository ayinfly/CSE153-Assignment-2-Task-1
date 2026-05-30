"""
Preprocessing utilities for onset-event symbolic music generation.

This file loads MIDI files, groups notes that start at the same time into
multi-note onset events, and saves train/test splits for event-based Markov modeling.
"""

from pathlib import Path
import pickle
import random

from miditoolkit import MidiFile


def find_midi_files(raw_dir):
    raw_dir = Path(raw_dir)
    midi_paths = list(raw_dir.rglob("*.mid")) + list(raw_dir.rglob("*.midi"))
    return sorted(midi_paths)


def extract_notes(path):
    midi = MidiFile(str(path))
    notes = []

    for inst in midi.instruments:
        if inst.is_drum:
            continue

        for note in inst.notes:
            duration = note.end - note.start

            if duration <= 0:
                continue

            notes.append({
                "pitch": int(note.pitch),
                "start": int(note.start),
                "end": int(note.end),
                "duration": int(duration),
                "velocity": int(note.velocity)
            })

    notes = sorted(notes, key=lambda x: (x["start"], x["pitch"]))
    return notes


def quantize_value(x, grid=120, min_value=0, max_value=1920):
    x = int(round(x / grid) * grid)
    x = max(min_value, x)
    x = min(max_value, x)
    return x


def midi_to_event_sequence(
    path,
    min_events=20,
    max_chord_size=8,
    quantize_grid=120,
    max_duration=1920,
    max_gap=1920
):
    notes = extract_notes(path)

    if len(notes) == 0:
        return None

    events_by_start = {}

    for note in notes:
        start = quantize_value(
            note["start"],
            grid=quantize_grid,
            min_value=0,
            max_value=10**12
        )

        duration = quantize_value(
            note["duration"],
            grid=quantize_grid,
            min_value=quantize_grid,
            max_value=max_duration
        )

        if start not in events_by_start:
            events_by_start[start] = []

        events_by_start[start].append({
            "pitch": note["pitch"],
            "duration": duration,
            "velocity": note["velocity"]
        })

    starts = sorted(events_by_start.keys())

    events = []
    prev_start = starts[0]

    for i, start in enumerate(starts):
        notes_at_start = sorted(
            events_by_start[start],
            key=lambda x: x["pitch"]
        )

        notes_at_start = notes_at_start[:max_chord_size]

        pitches = tuple(n["pitch"] for n in notes_at_start)
        durations = tuple(n["duration"] for n in notes_at_start)
        velocities = tuple(n["velocity"] for n in notes_at_start)

        if i == 0:
            gap = 0
        else:
            gap = start - prev_start

        gap = quantize_value(
            gap,
            grid=quantize_grid,
            min_value=0,
            max_value=max_gap
        )

        events.append({
            "gap": int(gap),
            "pitches": pitches,
            "durations": durations,
            "velocities": velocities
        })

        prev_start = start

    if len(events) < min_events:
        return None

    return {
        "path": str(path),
        "events": events,
        "num_events": len(events),
        "num_notes": sum(len(e["pitches"]) for e in events)
    }


def process_midi_files(
    midi_paths,
    max_files=None,
    min_events=20,
    max_chord_size=8,
    verbose=True
):
    if max_files is not None:
        midi_paths = midi_paths[:max_files]

    songs = []
    failed = []

    for i, path in enumerate(midi_paths):
        try:
            song = midi_to_event_sequence(
                path,
                min_events=min_events,
                max_chord_size=max_chord_size
            )

            if song is not None:
                songs.append(song)

        except Exception as e:
            failed.append((str(path), str(e)))

        if verbose and (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(midi_paths)} files | "
                f"usable: {len(songs)} | failed: {len(failed)}"
            )

    return songs, failed


def train_test_split_songs(songs, test_size=0.2, seed=0):
    songs = list(songs)

    random.seed(seed)
    random.shuffle(songs)

    split_idx = int((1 - test_size) * len(songs))

    train_songs = songs[:split_idx]
    test_songs = songs[split_idx:]

    return train_songs, test_songs


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_dataset(
    raw_dir,
    processed_dir,
    max_files=500,
    min_events=20,
    max_chord_size=8,
    test_size=0.2,
    seed=0
):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    midi_paths = find_midi_files(raw_dir)

    print("MIDI files found:", len(midi_paths))

    songs, failed = process_midi_files(
        midi_paths,
        max_files=max_files,
        min_events=min_events,
        max_chord_size=max_chord_size,
        verbose=True
    )

    train_songs, test_songs = train_test_split_songs(
        songs,
        test_size=test_size,
        seed=seed
    )

    save_pickle(train_songs, processed_dir / "train_songs.pkl")
    save_pickle(test_songs, processed_dir / "test_songs.pkl")
    save_pickle(failed, processed_dir / "failed_files.pkl")

    print("Usable songs:", len(songs))
    print("Train songs:", len(train_songs))
    print("Test songs:", len(test_songs))
    print("Failed files:", len(failed))

    return train_songs, test_songs, failed