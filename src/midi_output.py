"""
MIDI output utilities for event-based symbolic music generation.

This file converts generated onset events into playable MIDI files. Each event
can contain multiple simultaneous notes, which allows the model to generate
chord-like piano textures directly.
"""

from pathlib import Path

from miditoolkit import MidiFile, Instrument, Note


def save_event_midi(
    events,
    out_path,
    program=0,
    velocity=80,
    min_duration=60,
    max_duration=1920
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    midi = MidiFile()
    instrument = Instrument(program=program, is_drum=False, name="Generated Piano")

    current_time = 0

    for event in events:
        gap = max(0, int(event["gap"]))
        current_time += gap

        pitches = event["pitches"]
        durations = event["durations"]

        for pitch, duration in zip(pitches, durations):
            pitch = int(pitch)
            duration = int(duration)

            if not (21 <= pitch <= 108):
                continue

            duration = max(min_duration, duration)
            duration = min(max_duration, duration)

            note = Note(
                velocity=int(velocity),
                pitch=pitch,
                start=int(current_time),
                end=int(current_time + duration)
            )

            instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.dump(str(out_path))