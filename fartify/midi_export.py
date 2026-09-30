"""Minimal, dependency-free Standard MIDI File (format 0) writer.

We only need note-on/note-off events for the extracted melody, so rather
than pull in pretty_midi/mido, this writes raw MIDI bytes directly. This
keeps MIDI export working even in environments where installing extra
Python packages isn't convenient.
"""
from __future__ import annotations

from .notes import Note

TICKS_PER_BEAT = 480
TEMPO_US_PER_BEAT = 500_000  # 120 BPM, used purely as a fixed time base
TICKS_PER_SEC = TICKS_PER_BEAT * 1_000_000 / TEMPO_US_PER_BEAT  # 960 ticks/sec


def _vlq(value: int) -> bytes:
    """Encode an int as a MIDI variable-length quantity."""
    buf = [value & 0x7F]
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buf))


def _volume_to_velocity(volume_rms: float, notes: list[Note]) -> int:
    if not notes:
        return 96
    vols = [n.volume_rms for n in notes if n.volume_rms > 0]
    if not vols:
        return 96
    lo, hi = min(vols), max(vols)
    if hi - lo < 1e-9:
        return 96
    scaled = (volume_rms - lo) / (hi - lo)
    return int(max(1, min(127, round(40 + scaled * 87))))


def write_midi(notes: list[Note], path: str, track_name: str = "Fartify Melody") -> None:
    events: list[tuple[int, bytes]] = []  # (absolute_tick, midi_bytes)

    for note in notes:
        midi_num = int(round(max(0, min(127, note.midi_note))))
        velocity = _volume_to_velocity(note.volume_rms, notes)
        start_tick = int(round(note.start_sec * TICKS_PER_SEC))
        end_tick = int(round(note.end_sec * TICKS_PER_SEC))
        end_tick = max(end_tick, start_tick + 1)
        events.append((start_tick, bytes([0x90, midi_num, velocity])))  # note on
        events.append((end_tick, bytes([0x80, midi_num, 0])))  # note off

    events.sort(key=lambda e: (e[0], e[1][0] == 0x90))  # note-offs before note-ons at same tick

    track_data = bytearray()

    # Track name meta event
    name_bytes = track_name.encode("ascii", errors="replace")
    track_data += _vlq(0) + bytes([0xFF, 0x03, len(name_bytes)]) + name_bytes

    # Tempo meta event
    tempo_bytes = TEMPO_US_PER_BEAT.to_bytes(3, "big")
    track_data += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + tempo_bytes

    prev_tick = 0
    for abs_tick, midi_bytes in events:
        delta = max(0, abs_tick - prev_tick)
        track_data += _vlq(delta) + midi_bytes
        prev_tick = abs_tick

    # End of track
    track_data += _vlq(0) + bytes([0xFF, 0x2F, 0x00])

    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + TICKS_PER_BEAT.to_bytes(2, "big")
    track_chunk = b"MTrk" + len(track_data).to_bytes(4, "big") + bytes(track_data)

    with open(path, "wb") as f:
        f.write(header + track_chunk)
